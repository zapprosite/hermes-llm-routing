# agent/cloud_tier.py
import os
import sys
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from agent.voice_state import VoiceState
from agent.memory_governor import load_governor

_logger = logging.getLogger("hermes.cloud_tier")

@dataclass(frozen=True)
class CloudTierConfig:
    model: str
    provider: str
    base_url: str
    source: str

def _make_openai_client(base_url: str, *, api_key_env: str = "OPENAI_API_KEY"):
    from openai import AsyncOpenAI
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "not-needed")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

async def _openai_token_stream(
    client,
    model: str,
    messages: list,
    *,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if max_tokens is not None and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    stream = await client.chat.completions.create(**kwargs)
    iterator = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=5.0)
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        except StopAsyncIteration:
            break

async def _local_token_stream_with_fallback(
    client,
    model: str,
    messages: list,
    cloud: CloudTierConfig,
    state_callback=None,
    timeout_s: float = 3.0,
    *,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    stream = None
    try:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        create_task = client.chat.completions.create(**kwargs)
        stream = await asyncio.wait_for(create_task, timeout=timeout_s)
    except (asyncio.TimeoutError, Exception) as e:
        _logger.warning(f"Local model startup failed or timed out: {e}. Falling back to Cloud tier.")
        if state_callback:
            state_callback(VoiceState.THINKING_FALLBACK, f"Local startup failure: {e}")
            state_callback(VoiceState.ERROR_RECOVERY, "Local startup failure recovery")
        async for tok in _cloud_token_stream(cloud, messages, max_tokens=max_tokens):
            yield tok
        return

    iterator = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=5.0)
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        except StopAsyncIteration:
            break
        except (asyncio.TimeoutError, Exception) as e:
            _logger.warning(f"Local stream failed mid-generation: {e}. Falling back to Cloud tier.")
            if state_callback:
                state_callback(VoiceState.THINKING_FALLBACK, f"Local mid-stream failure: {e}")
                state_callback(VoiceState.ERROR_RECOVERY, "Local mid-stream failure recovery")
            async for tok in _cloud_token_stream(cloud, messages, max_tokens=max_tokens):
                yield tok
            break

def _load_runtime_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}

    try:
        cfg = load_config()
    except Exception:
        return {}

    return cfg if isinstance(cfg, dict) else {}

def _config_model_default(config: dict[str, Any]) -> str | None:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return None
    model = model_cfg.get("default")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None

def _config_model_field(config: dict[str, Any], field: str) -> str | None:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return None
    value = model_cfg.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None

def _resolve_cloud_tier(config: dict[str, Any]) -> CloudTierConfig:
    model = _config_model_default(config)
    provider = _config_model_field(config, "provider")
    base_url = _config_model_field(config, "base_url")

    if model and provider and base_url:
        if "127.0.0.1:4018" in base_url or "localhost:4018" in base_url:
            raise RuntimeError(
                "cloud tier resolved to dead :4018 endpoint from config.yaml; "
                "model.base_url must point at the provider endpoint"
            )
        return CloudTierConfig(
            model=model,
            provider=provider,
            base_url=base_url.rstrip("/"),
            source="config.yaml model.{default,provider,base_url}",
        )

    if not config:
        env_model = os.environ.get("HERMES_T2_MODEL", "").strip()
        env_provider = os.environ.get("HERMES_T2_PROVIDER", "").strip() or "openai"
        env_base_url = os.environ.get("HERMES_T2_BASE_URL", "").strip()
        if env_model and env_base_url:
            if "127.0.0.1:4018" in env_base_url or "localhost:4018" in env_base_url:
                raise RuntimeError("HERMES_T2_BASE_URL must not point at dead :4018")
            return CloudTierConfig(
                model=env_model,
                provider=env_provider,
                base_url=env_base_url.rstrip("/"),
                source="HERMES_T2_* env fallback",
            )

    missing = [
        name
        for name, value in (
            ("model.default", model),
            ("model.provider", provider),
            ("model.base_url", base_url),
        )
        if not value
    ]
    raise RuntimeError("cloud tier config incomplete: missing " + ", ".join(missing))

def _cloud_model_log_line(cloud: CloudTierConfig) -> str:
    return (
        "voice cloud tier resolved "
        f"model={cloud.model} "
        f"provider={cloud.provider} "
        f"base_url={cloud.base_url} "
        f"source={cloud.source}"
    )

_RUNTIME_CONFIG = _load_runtime_config()
LOCAL_BASE_URL = os.environ.get("HERMES_T1_BASE_URL", "http://127.0.0.1:8001/v1")
LOCAL_MODEL = os.environ.get("HERMES_T1_MODEL", "hermes-local")

def _anthropic_text_from_response(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "".join(parts)

# Cache de client + token provider para REUSO de conexão (keep-alive httpx).
# Rebuildar a cada turno forçava handshake TLS frio ao endpoint cloud. O client
# de build_anthropic_client reescreve o Authorization por-request via event hook,
# então o token segue fresco mesmo com o client memoizado.
_ANTHROPIC_CLIENT_CACHE: dict[str, Any] = {}
_MINIMAX_TOKEN_PROVIDER: Any = None


def _get_minimax_token_provider():
    global _MINIMAX_TOKEN_PROVIDER
    if _MINIMAX_TOKEN_PROVIDER is None:
        from hermes_cli.auth import build_minimax_oauth_token_provider
        _MINIMAX_TOKEN_PROVIDER = build_minimax_oauth_token_provider()
    return _MINIMAX_TOKEN_PROVIDER


def _get_anthropic_client(base_url: str):
    client = _ANTHROPIC_CLIENT_CACHE.get(base_url)
    if client is None:
        from agent.anthropic_adapter import build_anthropic_client
        client = build_anthropic_client(
            _get_minimax_token_provider(), base_url, timeout=120.0
        )
        _ANTHROPIC_CLIENT_CACHE[base_url] = client
    return client


async def _anthropic_token_stream(
    cloud: CloudTierConfig,
    messages: list,
    *,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    from agent.anthropic_adapter import build_anthropic_kwargs

    if cloud.provider != "minimax-oauth":
        raise RuntimeError(f"unsupported anthropic-compatible cloud provider: {cloud.provider}")

    client = _get_anthropic_client(cloud.base_url)
    kwargs = build_anthropic_kwargs(
        model=cloud.model,
        messages=messages,
        tools=None,
        max_tokens=max_tokens or load_governor(_RUNTIME_CONFIG).provider("minimax_m3").max_output_tokens,
        reasoning_config=None,
        base_url=cloud.base_url,
    )

    # Streaming incremental (SDK Anthropic messages.stream) bridgeado p/ asyncio
    # via fila: tokens cloud chegam cedo → melhor arbitragem do hedge e barge-in
    # mais rápido. Fallback fail-safe p/ create() não-streaming se o endpoint
    # MiniMax não suportar streaming (degrada sem quebrar a voz).
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _pump() -> None:
        try:
            with client.messages.stream(**kwargs) as stream:
                for delta in stream.text_stream:
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
        except Exception as exc:  # endpoint sem streaming, kwargs incompatíveis, etc.
            loop.call_soon_threadsafe(queue.put_nowait, ("__ERR__", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    pump_task = asyncio.create_task(asyncio.to_thread(_pump))
    streamed_any = False
    stream_error: Optional[Exception] = None
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__ERR__":
                stream_error = item[1]
                break
            streamed_any = True
            yield item
    finally:
        if not pump_task.done():
            pump_task.cancel()
        try:
            await pump_task
        except BaseException:
            pass

    # Streaming falhou sem emitir nada → fallback p/ create() não-streaming.
    if stream_error is not None and not streamed_any:
        _logger.warning(
            "Anthropic streaming indisponível (%s); fallback non-stream.", stream_error
        )
        response = await asyncio.to_thread(client.messages.create, **kwargs)
        text = _anthropic_text_from_response(response)
        if text:
            yield text

def _cloud_token_stream(
    cloud: CloudTierConfig,
    messages: list,
    *,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    if max_tokens is None:
        max_tokens = load_governor(_RUNTIME_CONFIG).provider("minimax_m3").max_output_tokens
    if cloud.provider == "minimax-oauth" or "/anthropic" in cloud.base_url:
        return _anthropic_token_stream(cloud, messages, max_tokens=max_tokens)
    client = _make_openai_client(cloud.base_url, api_key_env="HERMES_T2_API_KEY")
    return _openai_token_stream(client, cloud.model, messages, max_tokens=max_tokens)
