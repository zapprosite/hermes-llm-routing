"""Pytest fixtures compartilhadas para smoke tests do kit."""
import pytest


@pytest.fixture
def skip_if_no_gpu():
    """Skip test se GPU nao disponivel."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU nao disponivel")
    except ImportError:
        pytest.skip("torch nao instalado")


@pytest.fixture
def hermes_home():
    """Path do HERMES_HOME (default ~/.hermes)."""
    import os
    from pathlib import Path
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
