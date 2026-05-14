import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SUPPORTED_LANGS = {"es", "en"}
TARGET_SAMPLE_RATE = 16000
TEMP_DIR = Path(tempfile.gettempdir()) / "traductor_voz_ai"
TEMP_DIR.mkdir(exist_ok=True)
TEMP_TTL_SECONDS = 3600

_model: WhisperModel | None = None
_client: ElevenLabs | None = None


def get_model() -> WhisperModel:
    """Lazy-load the Whisper model so import-time is cheap and tests can patch it."""
    global _model
    if _model is None:
        log.info("Cargando Whisper '%s' (device=%s, compute=%s)", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _model


def get_client() -> ElevenLabs:
    """Lazy-load the ElevenLabs client. Raises if the API key is missing."""
    global _client
    if _client is None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta ELEVENLABS_API_KEY. Copiá env.example a .env y configurá tu clave."
            )
        _client = ElevenLabs(api_key=api_key)
    return _client


def cleanup_old_temp_files(ttl_seconds: int = TEMP_TTL_SECONDS) -> int:
    """Delete TTS mp3 files older than ttl_seconds. Returns the count removed."""
    cutoff = time.time() - ttl_seconds
    removed = 0
    for f in TEMP_DIR.glob("*.mp3"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def transcribe(audio: str | np.ndarray, idioma_entrada: str, *, model: WhisperModel | None = None) -> tuple[str, str]:
    """Returns (detected_language, text). Accepts a filepath or a numpy waveform at 16kHz."""
    if model is None:
        model = get_model()
    lang = None if idioma_entrada == "auto" else idioma_entrada
    segments, info = model.transcribe(audio, language=lang, vad_filter=True)
    texto = " ".join(seg.text for seg in segments).strip()
    detected = info.language if lang is None else lang
    if detected not in SUPPORTED_LANGS:
        log.warning("Idioma detectado '%s' no soportado, usando 'es'", detected)
        detected = "es"
    return detected, texto


def translate_text(text: str, source: str, target: str) -> str:
    return GoogleTranslator(source=source, target=target).translate(text)


def synthesize(text: str, *, client: ElevenLabs | None = None) -> Path:
    """Generate TTS audio and return the path to the saved mp3."""
    if client is None:
        client = get_client()
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
    )
    out_path = TEMP_DIR / f"tts_{int(time.time() * 1000)}.mp3"
    with out_path.open("wb") as out_file:
        for chunk in audio_stream:
            if chunk:
                out_file.write(chunk)
    return out_path


def traducir(audio_path: str, idioma_entrada: str):
    if not audio_path:
        raise gr.Error("Grabá audio antes de traducir.")

    cleanup_old_temp_files()

    try:
        idioma_in, texto = transcribe(audio_path, idioma_entrada)
    except Exception as e:
        log.exception("Error en transcripción")
        raise gr.Error(f"No pude transcribir el audio: {e}") from e

    if not texto:
        raise gr.Error("No detecté ninguna palabra en el audio.")

    idioma_salida = "en" if idioma_in == "es" else "es"

    try:
        traduccion = translate_text(texto, idioma_in, idioma_salida)
    except Exception as e:
        log.exception("Error en traducción")
        raise gr.Error(f"Falló la traducción: {e}") from e

    try:
        out_path = synthesize(traduccion)
    except Exception as e:
        log.exception("Error generando voz")
        raise gr.Error(f"Falló la generación de voz (ElevenLabs): {e}") from e

    return texto, traduccion, str(out_path)


# ---------- Live streaming mode ----------

def _resample_to_16k(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Cheap linear resample. Good enough for Whisper input."""
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = audio.shape[0] / sample_rate
    target_len = int(duration * TARGET_SAMPLE_RATE)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    indices = np.linspace(0, audio.shape[0] - 1, target_len)
    return audio[indices.astype(np.int64)].astype(np.float32)


def live_transcribe(new_chunk, state, idioma_entrada):
    """Streaming callback: accumulates chunks and updates transcription live."""
    if new_chunk is None:
        return state, ""

    sample_rate, data = new_chunk
    data = data.astype(np.float32)
    if np.max(np.abs(data)) > 1.0:
        data = data / 32768.0  # int16 → float32
    chunk_16k = _resample_to_16k(sample_rate, data)

    if state is None:
        state = np.zeros(0, dtype=np.float32)
    state = np.concatenate([state, chunk_16k])

    # Cap accumulated audio at 30s to keep latency bounded
    max_samples = TARGET_SAMPLE_RATE * 30
    if state.shape[0] > max_samples:
        state = state[-max_samples:]

    try:
        _, texto = transcribe(state, idioma_entrada)
    except Exception as e:
        log.exception("Error en live transcribe")
        return state, f"[error: {e}]"
    return state, texto


def reset_live_state():
    return None, ""


# ---------- UI ----------

with gr.Blocks(title="Asistente Bilingüe en Tiempo Real 🌍") as interface:
    gr.Markdown("# Asistente Bilingüe en Tiempo Real 🌍\nHablá en español o inglés. Escuchá la traducción con voz realista.")

    with gr.Tabs():
        with gr.Tab("🎤 Push-to-talk"):
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Hablá aquí")
                lang_in = gr.Radio(
                    ["auto", "es", "en"],
                    label="Idioma de entrada",
                    value="auto",
                    info="'auto' detecta el idioma con Whisper.",
                )
            btn = gr.Button("Traducir", variant="primary")
            transcripcion = gr.Textbox(label="📝 Transcripción")
            traduccion = gr.Textbox(label="🌐 Traducción")
            audio_out = gr.Audio(label="🔊 Traducción hablada")
            btn.click(traducir, [audio_in, lang_in], [transcripcion, traduccion, audio_out])

        with gr.Tab("📡 Modo continuo (transcripción en vivo)"):
            gr.Markdown(
                "Activá el micrófono y empezá a hablar. La transcripción se actualiza mientras hablás. "
                "Para traducir + sintetizar voz, usá la pestaña anterior."
            )
            live_lang = gr.Radio(["auto", "es", "en"], label="Idioma", value="auto")
            live_audio = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="🎙️ Streaming")
            live_state = gr.State()
            live_text = gr.Textbox(label="📝 Transcripción en vivo", lines=4)
            clear_btn = gr.Button("Limpiar")
            live_audio.stream(live_transcribe, [live_audio, live_state, live_lang], [live_state, live_text])
            clear_btn.click(reset_live_state, outputs=[live_state, live_text])

if __name__ == "__main__":
    interface.launch()
