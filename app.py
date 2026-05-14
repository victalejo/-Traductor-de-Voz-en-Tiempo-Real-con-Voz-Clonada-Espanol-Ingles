import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError(
        "Falta ELEVENLABS_API_KEY. Copiá env.example a .env y configurá tu clave."
    )

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SUPPORTED_LANGS = {"es", "en"}
TEMP_DIR = Path(tempfile.gettempdir()) / "traductor_voz_ai"
TEMP_DIR.mkdir(exist_ok=True)
TEMP_TTL_SECONDS = 3600

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
log.info("Cargando Whisper '%s' (device=%s, compute=%s)", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)


def _cleanup_old_temp_files() -> None:
    cutoff = time.time() - TEMP_TTL_SECONDS
    for f in TEMP_DIR.glob("*.mp3"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _transcribe(audio_path: str, idioma_entrada: str) -> tuple[str, str]:
    """Returns (detected_language, text)."""
    lang = None if idioma_entrada == "auto" else idioma_entrada
    segments, info = model.transcribe(audio_path, language=lang, vad_filter=True)
    texto = " ".join(seg.text for seg in segments).strip()
    detected = info.language if lang is None else lang
    if detected not in SUPPORTED_LANGS:
        log.warning("Idioma detectado '%s' no soportado, usando 'es'", detected)
        detected = "es"
    return detected, texto


def traducir(audio_path: str, idioma_entrada: str):
    if not audio_path:
        raise gr.Error("Grabá audio antes de traducir.")

    _cleanup_old_temp_files()

    try:
        idioma_in, texto = _transcribe(audio_path, idioma_entrada)
    except Exception as e:
        log.exception("Error en transcripción")
        raise gr.Error(f"No pude transcribir el audio: {e}") from e

    if not texto:
        raise gr.Error("No detecté ninguna palabra en el audio.")

    idioma_salida = "en" if idioma_in == "es" else "es"

    try:
        traduccion = GoogleTranslator(source=idioma_in, target=idioma_salida).translate(texto)
    except Exception as e:
        log.exception("Error en traducción")
        raise gr.Error(f"Falló la traducción: {e}") from e

    try:
        audio_stream = client.text_to_speech.convert(
            text=traduccion,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
        )
        out_path = TEMP_DIR / f"tts_{int(time.time() * 1000)}.mp3"
        with out_path.open("wb") as out_file:
            for chunk in audio_stream:
                if chunk:
                    out_file.write(chunk)
    except Exception as e:
        log.exception("Error generando voz")
        raise gr.Error(f"Falló la generación de voz (ElevenLabs): {e}") from e

    return texto, traduccion, str(out_path)


interface = gr.Interface(
    fn=traducir,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Hablá aquí"),
        gr.Radio(
            ["auto", "es", "en"],
            label="Idioma de entrada",
            value="auto",
            info="'auto' detecta el idioma con Whisper.",
        ),
    ],
    outputs=[
        gr.Textbox(label="📝 Transcripción"),
        gr.Textbox(label="🌐 Traducción"),
        gr.Audio(label="🔊 Traducción hablada"),
    ],
    title="Asistente Bilingüe en Tiempo Real 🌍",
    description="Hablá en español o inglés. Escuchá la traducción con voz realista al instante.",
)

if __name__ == "__main__":
    interface.launch()
