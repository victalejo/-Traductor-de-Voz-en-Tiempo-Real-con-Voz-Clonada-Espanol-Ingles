import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import whisper
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

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
SUPPORTED_LANGS = {"es", "en"}
TEMP_DIR = Path(tempfile.gettempdir()) / "traductor_voz_ai"
TEMP_DIR.mkdir(exist_ok=True)
TEMP_TTL_SECONDS = 3600  # files older than 1h are removed on the next call

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
model = whisper.load_model(WHISPER_MODEL)


def _cleanup_old_temp_files() -> None:
    cutoff = time.time() - TEMP_TTL_SECONDS
    for f in TEMP_DIR.glob("*.mp3"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _detect_or_use(audio_path: str, idioma_entrada: str) -> tuple[str, str]:
    """Returns (input_lang, transcribed_text). Falls back to manual choice if auto fails."""
    if idioma_entrada != "auto":
        result = model.transcribe(audio_path, language=idioma_entrada)
        return idioma_entrada, result["text"].strip()

    # Whisper's auto detection: load 30s, detect language, then transcribe
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    if detected not in SUPPORTED_LANGS:
        # Default to español if Whisper detected something we don't translate
        log.warning("Idioma detectado '%s' no soportado, usando 'es'", detected)
        detected = "es"
    result = model.transcribe(audio_path, language=detected)
    return detected, result["text"].strip()


def traducir(audio_path: str, idioma_entrada: str):
    if not audio_path:
        raise gr.Error("Grabá audio antes de traducir.")

    _cleanup_old_temp_files()

    try:
        idioma_in, texto = _detect_or_use(audio_path, idioma_entrada)
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
