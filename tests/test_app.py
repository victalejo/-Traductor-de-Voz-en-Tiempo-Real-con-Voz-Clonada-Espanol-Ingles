import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app


def _fake_model(text="hola mundo", language="es"):
    """Build a mock that mimics WhisperModel.transcribe()."""
    model = MagicMock()
    segments = [SimpleNamespace(text=text)]
    info = SimpleNamespace(language=language)
    model.transcribe.return_value = (segments, info)
    return model


# ---------- cleanup ----------

def test_cleanup_removes_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    old = tmp_path / "old.mp3"
    new = tmp_path / "new.mp3"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    old_time = time.time() - 7200
    os.utime(old, (old_time, old_time))

    removed = app.cleanup_old_temp_files(ttl_seconds=3600)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_cleanup_handles_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    assert app.cleanup_old_temp_files() == 0


# ---------- transcribe ----------

def test_transcribe_with_explicit_language():
    model = _fake_model(text=" hola ", language="es")
    lang, texto = app.transcribe("fake.wav", "es", model=model)
    assert lang == "es"
    assert texto == "hola"
    assert model.transcribe.call_args.kwargs["language"] == "es"


def test_transcribe_auto_detects_language():
    model = _fake_model(text="hello", language="en")
    lang, texto = app.transcribe("fake.wav", "auto", model=model)
    assert lang == "en"
    assert texto == "hello"
    assert model.transcribe.call_args.kwargs["language"] is None


def test_transcribe_falls_back_when_language_not_supported():
    model = _fake_model(text="bonjour", language="fr")
    lang, _ = app.transcribe("fake.wav", "auto", model=model)
    assert lang == "es"


# ---------- translate ----------

def test_translate_text_invokes_google_translator():
    with patch("app.GoogleTranslator") as gt:
        gt.return_value.translate.return_value = "hello world"
        result = app.translate_text("hola mundo", "es", "en")
    assert result == "hello world"
    gt.assert_called_once_with(source="es", target="en")


# ---------- synthesize ----------

def test_synthesize_writes_mp3(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    client = MagicMock()
    client.text_to_speech.convert.return_value = [b"abc", b"def"]
    path = app.synthesize("hello", client=client)
    assert path.exists()
    assert path.read_bytes() == b"abcdef"
    assert path.suffix == ".mp3"


# ---------- traducir orchestrator ----------

def test_traducir_raises_on_empty_audio():
    with pytest.raises(app.gr.Error):
        app.traducir(None, "auto")


def test_traducir_raises_on_empty_transcription(monkeypatch):
    monkeypatch.setattr(app, "transcribe", lambda *a, **kw: ("es", ""))
    monkeypatch.setattr(app, "cleanup_old_temp_files", lambda *a, **kw: 0)
    with pytest.raises(app.gr.Error):
        app.traducir("fake.wav", "auto")


def test_traducir_happy_path(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "transcribe", lambda *a, **kw: ("es", "hola"))
    monkeypatch.setattr(app, "translate_text", lambda *a, **kw: "hi")
    out_file = tmp_path / "out.mp3"
    out_file.write_bytes(b"audio")
    monkeypatch.setattr(app, "synthesize", lambda *a, **kw: out_file)
    monkeypatch.setattr(app, "cleanup_old_temp_files", lambda *a, **kw: 0)

    texto, traduccion, path = app.traducir("fake.wav", "auto")
    assert texto == "hola"
    assert traduccion == "hi"
    assert path == str(out_file)


def test_traducir_wraps_transcribe_errors(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("whisper down")

    monkeypatch.setattr(app, "transcribe", boom)
    monkeypatch.setattr(app, "cleanup_old_temp_files", lambda *a, **kw: 0)
    with pytest.raises(app.gr.Error):
        app.traducir("fake.wav", "auto")


# ---------- streaming helpers ----------

def test_resample_passthrough_at_16k():
    audio = np.ones(16000, dtype=np.float32)
    out = app._resample_to_16k(16000, audio)
    assert out.shape[0] == 16000


def test_resample_downsamples_correctly():
    audio = np.ones(48000, dtype=np.float32)
    out = app._resample_to_16k(48000, audio)
    assert abs(out.shape[0] - 16000) <= 1


def test_resample_handles_stereo():
    audio = np.ones((48000, 2), dtype=np.float32)
    out = app._resample_to_16k(48000, audio)
    assert out.ndim == 1


def test_live_transcribe_accumulates_state(monkeypatch):
    monkeypatch.setattr(app, "transcribe", lambda audio, lang, model=None: ("es", "parcial"))
    chunk = (16000, np.ones(1600, dtype=np.float32))
    state, text = app.live_transcribe(chunk, None, "auto")
    assert state.shape[0] == 1600
    assert text == "parcial"

    state2, _ = app.live_transcribe(chunk, state, "auto")
    assert state2.shape[0] == 3200


def test_live_transcribe_returns_empty_on_none():
    state, text = app.live_transcribe(None, None, "auto")
    assert state is None
    assert text == ""


def test_reset_live_state():
    state, text = app.reset_live_state()
    assert state is None
    assert text == ""
