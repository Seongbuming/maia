import os
import gradio as gr
import tempfile
import base64
from google.cloud import texttospeech

from utils.model import Model

class GoogleTTS(Model):
    def __init__(
        self,
        api_key: str = "",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_TTS_API_KEY", "")
        self.setup_interface(self.synthesize, self.get_inputs(), self.get_outputs())
        
    def synthesize(
        self,
        text: str,
        language_code: str = "en-US",
    ):
        audio_file = tempfile.NamedTemporaryFile(delete=True, suffix=".mp3")

        client = texttospeech.TextToSpeechClient(
            client_options={ "api_key": self.api_key }
        )

        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=f"en-US-Wavenet-G",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        
        with open(audio_file.name, "wb") as out:
            out.write(response.audio_content)

        speech_base64 = self.__audio_to_base64(audio_file.name)

        return speech_base64
        
    def get_inputs(self):
        return [
            gr.components.Textbox(lines=2, label="Input", placeholder=None),
        ]

    def get_outputs(self):
        return [
            gr.components.Audio(
                label="Speech",
                type="filepath",
            ),
        ]
    
    def __audio_to_base64(self, audio_filepath: str) -> str:
        with open(audio_filepath, "rb") as audio_file:
            audio_encoded = base64.b64encode(audio_file.read())
        return audio_encoded.decode("utf-8")
