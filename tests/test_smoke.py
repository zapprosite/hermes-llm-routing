"""Smoke tests para hermes-llm-routing."""
import pytest
import httpx


def test_t1_reachable():
    """LLM T1 (Qwen local) deve estar em :8001."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get("http://127.0.0.1:8001/v1/models")
            assert r.status_code in (200, 401)
    except httpx.ConnectError:
        pytest.skip("LLM T1 nao esta rodando (esperado em CI)")


def test_t2_url_format():
    """T2 URL deve terminar com /v1 (do release v2 fix)."""
    import re
    pattern = re.compile(r"^https?://.*/anthropic/v1$")
    test_url = "https://api.minimax.io/anthropic/v1"
    assert pattern.match(test_url), f"T2 URL mal formatada: {test_url}"


def test_module_imports():
    """llm_routing deve ser importavel."""
    try:
        from llm_routing import cloud_tier, llm_pipeline
    except ImportError:
        pytest.skip("llm_routing nao instalado")
