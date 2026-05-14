import os
import tempfile

import gradio as gr
import whisper
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError(
        "Falta ELEVENLABS_API_KEY. Copiá env.example a .env y configurá tu clave."
    )

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
model = whisper.load_model(WHISPER_MODEL)


def traducir(audio_path: str, idioma_entrada: str):
    if not audio_path:
        return "", "", None

    texto = model.transcribe(audio_path, language=idioma_entrada)["text"].strip()
    idioma_salida = "en" if idioma_entrada == "es" else "es"
    traduccion = GoogleTranslator(source=idioma_entrada, target=idioma_salida).translate(texto)

    audio_stream = client.text_to_speech.convert(
        text=traduccion,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_file:
        for chunk in audio_stream:
            if chunk:
                out_file.write(chunk)
        return texto, traduccion, out_file.name


interface = gr.Interface(
    fn=traducir,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Hablá aquí"),
        gr.Radio(["es", "en"], label="Idioma que estás hablando", value="es"),
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
