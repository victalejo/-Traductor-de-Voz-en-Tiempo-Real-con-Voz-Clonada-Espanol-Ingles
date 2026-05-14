"""Shared fixtures: stub out heavy modules so `import app` is cheap in tests."""
import os
import sys
import types
from unittest.mock import MagicMock

os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Install a stub module so `import name` works without the real dependency."""
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub elevenlabs (the real package can be slow / require network)
elevenlabs_pkg = _install_stub("elevenlabs")
elevenlabs_client_mod = _install_stub("elevenlabs.client", {"ElevenLabs": MagicMock})
elevenlabs_pkg.client = elevenlabs_client_mod

# Stub faster_whisper so model loading never happens at import time
_install_stub("faster_whisper", {"WhisperModel": MagicMock})

# Stub deep_translator
_install_stub("deep_translator", {"GoogleTranslator": MagicMock})

# Stub gradio with the minimum surface app.py uses at import time
gradio_stub = types.ModuleType("gradio")


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _DummyComponent(_DummyCtx):
    def __init__(self, *a, **kw):
        pass

    def click(self, *a, **kw):
        pass

    def stream(self, *a, **kw):
        pass


gradio_stub.Blocks = lambda *a, **kw: _DummyComponent()
gradio_stub.Tabs = lambda *a, **kw: _DummyComponent()
gradio_stub.Tab = lambda *a, **kw: _DummyComponent()
gradio_stub.Row = lambda *a, **kw: _DummyComponent()
gradio_stub.Markdown = lambda *a, **kw: _DummyComponent()
gradio_stub.Audio = lambda *a, **kw: _DummyComponent()
gradio_stub.Radio = lambda *a, **kw: _DummyComponent()
gradio_stub.Textbox = lambda *a, **kw: _DummyComponent()
gradio_stub.Button = lambda *a, **kw: _DummyComponent()
gradio_stub.State = lambda *a, **kw: _DummyComponent()


class _GrError(Exception):
    pass


gradio_stub.Error = _GrError
sys.modules["gradio"] = gradio_stub
