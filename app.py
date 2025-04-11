
import gradio as gr
import whisper
import tempfile
from deep_translator import GoogleTranslator
from elevenlabs import generate, set_api_key

# CONFIG
set_api_key("TU_API_KEY")  # Reemplazar con tu API de ElevenLabs
VOICE_NAME = "Rachel"  # Voz de prueba de ElevenLabs
model = whisper.load_model("base")

# FUNCIONES
def traducir(audio, idioma_entrada):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio)
        temp_audio_path = f.name

    texto = model.transcribe(temp_audio_path, language=idioma_entrada)["text"]
    idioma_salida = "en" if idioma_entrada == "es" else "es"
    traduccion = GoogleTranslator(source=idioma_entrada, target=idioma_salida).translate(texto)

    audio_out = generate(
        text=traduccion,
        voice=VOICE_NAME,
        model="eleven_multilingual_v2"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_file:
        out_file.write(audio_out)
        return texto, traduccion, out_file.name

# INTERFAZ
interface = gr.Interface(
    fn=traducir,
    inputs=[
        gr.Audio(source="microphone", type="binary", label="🎙️ Hablá aquí"),
        gr.Radio(["es", "en"], label="Idioma que estás hablando", value="es")
    ],
    outputs=[
        gr.Textbox(label="📝 Transcripción"),
        gr.Textbox(label="🌐 Traducción"),
        gr.Audio(label="🔊 Traducción hablada")
    ],
    title="Asistente Bilingüe en Tiempo Real 🌍",
    description="Hablá en español o inglés. Escuchá la traducción con voz realista al instante."
)

interface.launch()
