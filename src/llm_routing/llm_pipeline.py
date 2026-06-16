"""LLM pipeline and speculative hedge runner for the voice agent."""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from agent.voice_state import VoiceState
from agent.barge_in import trigger_barge_in
from agent.hedge_voice import HedgeConfig, hedge_generate, nomic_cosine_arbiter
from agent.t0_skills_router import T0Router, Tier
from agent.cloud_tier import (
    _resolve_cloud_tier,
    _cloud_token_stream,
    _openai_token_stream,
    _local_token_stream_with_fallback,
    _cloud_model_log_line,
    _make_openai_client,
    _RUNTIME_CONFIG,
    LOCAL_BASE_URL,
    LOCAL_MODEL,
)
from agent.memory_governor import (
    estimate_messages_tokens as _estimate_messages_tokens,
    load_governor,
    messages_have_tools,
)

_logger = logging.getLogger("hermes.voice.llm_pipeline")


def _last_user_text(chat_ctx: Any) -> str:
    for message in reversed(chat_ctx.messages()):
        if message.role == "user" and getattr(message, "text_content", None):
            return message.text_content.strip()
    return ""


def estimate_messages_tokens(messages: list) -> int:
    """Compatibility wrapper for tests and callers; governor owns estimation."""
    return _estimate_messages_tokens(messages)

def get_operational_context(compact: bool = False) -> str:
    import subprocess
    import os
    
    summary_parts = []
    
    # 1. Project directory
    cwd = os.getenv("TERMINAL_CWD", os.getcwd())
    summary_parts.append(f"Projeto: {os.path.basename(cwd)}")
    
    # 2. Current git branch
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, text=True, timeout=1.5
        ).strip()
        summary_parts.append(f"Branch: {branch}")
    except Exception:
        pass
        
    # 3. Active voice services
    try:
        cp = subprocess.run(
            ["systemctl", "--user", "is-active", "hermes-livekit-agent.service"],
            capture_output=True, text=True, timeout=1.5
        )
        svc_active = "ativo" if cp.stdout.strip() == "active" else "inativo"
        summary_parts.append(f"Serviço de voz: {svc_active}")
    except Exception:
        pass
        
    # 4. System health snapshot
    try:
        import psutil
        import shutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = shutil.disk_usage("/").percent
        summary_parts.append(f"Health: CPU={cpu}%, MEM={mem}%, DISCO={disk}%")
    except Exception:
        pass

    # 5. Last commit / task
    if not compact:
        try:
            last_commit = subprocess.check_output(
                ["git", "log", "-1", "--oneline"],
                cwd=cwd, text=True, timeout=1.5
            ).strip()
            summary_parts.append(f"Última tarefa/commit: {last_commit}")
        except Exception:
            pass
        
    return " Memória Operacional:\n" + "\n".join(f"- {part}" for part in summary_parts)


def build_hedge_runner(config: dict[str, Any] | None = None):
    """Retorna uma coroutine `run(messages, on_token, on_bargein) -> HedgeOutcome`."""
    config = config or _RUNTIME_CONFIG
    router = T0Router.from_config(config)
    hcfg = HedgeConfig.from_config(config)
    local_client = _make_openai_client(LOCAL_BASE_URL)
    cloud = _resolve_cloud_tier(config)
    governor = load_governor(config)
    qwen_budget = governor.provider("qwen_voice_fast")
    minimax_budget = governor.provider("minimax_m3")
    _logger.info(_cloud_model_log_line(cloud))

    async def run(
        utterance: str,
        messages: list,
        on_token=None,
        on_bargein=None,
        state_callback=None,
        interrupted_event=None,
        state_machine=None,
    ):
        decision = router.route(utterance, context={"is_voice": True})

        # Importa os construtores de prompt do Jarvis
        from hermes_voice.persona.jarvis import build_system_prompt, BASE_SYSTEM_PROMPT, BRAIN_PRIORITY_PROMPT, VOICE_LATENCY_PROMPT
        
        local_sys = f"{BASE_SYSTEM_PROMPT}\n\n{BRAIN_PRIORITY_PROMPT}\n\n{VOICE_LATENCY_PROMPT}\n\n{get_operational_context(compact=True)}"
        cloud_sys = f"{build_system_prompt(include_soul=True)}\n\n{get_operational_context(compact=False)}"

        def inject_sys(msgs, sys_content):
            new_msgs = list(msgs)
            sys_idx = -1
            for idx, m in enumerate(new_msgs):
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_idx = idx
                    break
            if sys_idx != -1:
                new_msgs[sys_idx] = {"role": "system", "content": sys_content}
            else:
                new_msgs.insert(0, {"role": "system", "content": sys_content})
            return new_msgs

        local_messages = inject_sys(messages, local_sys)
        cloud_messages = inject_sys(messages, cloud_sys)
        local_messages, local_fit = governor.fit_messages("qwen_voice_fast", local_messages)
        cloud_messages, cloud_fit = governor.fit_messages("minimax_m3", cloud_messages)
        governor.log_decision(local_fit)
        governor.log_decision(cloud_fit)

        if local_fit.action == "escalate":
            decision.tier = Tier.CLOUD.value
            decision.reason = (
                "Forçado CLOUD pelo Memory Governor: "
                f"tokens={local_fit.prompt_tokens_estimated} "
                f"budget={local_fit.budget_total} "
                f"tools={messages_have_tools(messages)}"
            )
        elif decision.tier == Tier.LOCAL.value and local_fit.action == "compact":
            decision.reason = (
                f"{decision.reason}; Memory Governor compactou contexto local "
                f"tokens={local_fit.prompt_tokens_estimated}/{local_fit.budget_total}"
            )

        if decision.tier == Tier.CLOUD.value:
            if state_machine:
                state_machine.deep_analysis = True
            if state_callback:
                state_callback(VoiceState.THINKING_FALLBACK, "Routing to CLOUD tier (MiniMax-M3)")
            async for tok in _cloud_token_stream(
                cloud,
                cloud_messages,
                max_tokens=minimax_budget.max_output_tokens,
            ):
                if on_token:
                    on_token(tok, "cloud")
            return decision

        if decision.tier == Tier.LOCAL.value:
            if state_machine:
                state_machine.deep_analysis = False
            if state_callback:
                state_callback(VoiceState.THINKING_LOCAL, "Routing to LOCAL tier (Qwen local_fast)")
            async for tok in _local_token_stream_with_fallback(
                local_client,
                LOCAL_MODEL,
                local_messages,
                cloud,
                state_callback,
                timeout_s=hcfg.local_timeout_s,
                max_tokens=qwen_budget.max_output_tokens,
            ):
                if on_token:
                    on_token(tok, "local")
            return decision

        if state_callback:
            state_callback(VoiceState.THINKING_LOCAL, "Routing to HEDGE parallel tier")

        if state_machine:
            state_machine.deep_analysis = False

        return await hedge_generate(
            _openai_token_stream(
                local_client,
                LOCAL_MODEL,
                local_messages,
                max_tokens=qwen_budget.max_output_tokens,
            ),
            _cloud_token_stream(
                cloud,
                cloud_messages,
                max_tokens=minimax_budget.max_output_tokens,
            ),
            arbiter=nomic_cosine_arbiter(threshold=0.75),
            on_token=on_token,
            on_bargein=on_bargein,
            cfg=hcfg,
            interrupted_event=interrupted_event,
        )

    return run


async def run_llm_pipeline(
    chat_ctx: Any,
    runner: Any,
    state_machine: Any,
    lk_session: Any,
):
    """Executes the LLM pipeline and yields tokens asynchronously."""
    messages, _ = chat_ctx.to_provider_format("openai", inject_dummy_user_message=False)
    utterance = _last_user_text(chat_ctx)
    token_queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
    interrupted = asyncio.Event()

    wake_turn_id = state_machine.get_current_wake_turn_id()
    _logger.info("llm_request_started utterance=%r wake_turn_id=%s", utterance, wake_turn_id)
    if state_machine._redis is not None:
        try:
            import time
            state_machine._redis.hset(
                f"hermes:voice:turn:{wake_turn_id}",
                mapping={
                    "llm": "1",
                    "turn_llm": "1",
                    "turn_llm_start_ts": f"{time.time():.6f}",
                }
            )
        except Exception as e:
            _logger.debug(f"Failed to log LLM start in Redis: {e}")

    def _on_bargein() -> None:
        trigger_barge_in(state_machine, lk_session, interrupted)
        token_queue.put_nowait(None)

    async def _produce() -> None:
        try:
            state_machine.transition_to(VoiceState.ROUTING, "Starting llm_node routing")
            first_token_logged = False

            def on_token_wrapper(token, tier):
                nonlocal first_token_logged
                if not first_token_logged:
                    first_token_logged = True
                    w_id = state_machine.get_current_wake_turn_id()
                    _logger.info("llm_response_started tier=%s wake_turn_id=%s", tier, w_id)
                    if state_machine._redis is not None:
                        try:
                            state_machine._redis.hset(f"hermes:voice:turn:{w_id}", "llm", "1")
                        except Exception as e:
                            _logger.debug(f"Failed to log LLM response in Redis: {e}")
                token_queue.put_nowait(token)

            await runner(
                utterance,
                messages,
                on_token=on_token_wrapper,
                on_bargein=_on_bargein,
                state_callback=state_machine.transition_to,
                interrupted_event=interrupted,
                state_machine=state_machine,
            )
        except asyncio.CancelledError:
            import time
            if getattr(state_machine, "bargein_start_time", None) is not None:
                latency = (time.time() - state_machine.bargein_start_time) * 1000
                _logger.info(f"Barge-in LLM cancellation latency: {latency:.2f} ms")
            raise
        except BaseException as exc:
            error_str = str(exc)
            state_machine.transition_to(VoiceState.ERROR_RECOVERY, f"Runner failed: {exc}")
            # Record error in Redis and immediately return to IDLE.
            # We do NOT finalize here for hard failures (CancelledError propagates above).
            # For recoverable runner errors, finalize and recover.
            state_machine.finalize_turn("llm_runner_error_recovery", error=error_str)
            token_queue.put_nowait(exc)
        finally:
            token_queue.put_nowait(None)

    task = asyncio.create_task(_produce(), name="hedge.llm_node.produce")
    try:
        while not interrupted.is_set():
            item = await token_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()

        while not token_queue.empty():
            token_queue.get_nowait()

        if state_machine.current_state in (
            VoiceState.ROUTING,
            VoiceState.THINKING_LOCAL,
            VoiceState.THINKING_FALLBACK,
        ):
            # LLM finished but TTS never started (interrupted before speaking).
            # finalize_turn() resets to IDLE and updates last_transition.
            state_machine.finalize_turn(
                "llm_completed_no_tts: turn finalized before speaking"
            )
        elif state_machine.current_state in (
            VoiceState.INTERRUPTED,
            VoiceState.CANCELLED,
        ):
            # Interrupted/cancelled mid-LLM: finalize to IDLE.
            state_machine.finalize_turn(
                f"llm_interrupted: state={state_machine.current_state.value}"
            )
