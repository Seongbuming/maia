import gc
import gradio as gr
import torch
import whisperx

from utils.model import Model

class WhisperX(Model):
    def __init__(
        self,
        device: str = "cpu",
        device_index: int = 0,
        model_name: str = "large-v2",
        compute_type: str = "float16",  # reduce if low on GPU mem
        batch_size: int = 16,
    ):
        # 1. Transcribe with original whisper (batched)
        self.device = device
        self.device_index = device_index
        self.model_name = model_name
        self.compute_type = compute_type
        self.batch_size = batch_size

        self.setup_interface(self.transcribe, self.get_inputs(), self.get_outputs())
        
    def transcribe(
        self,
        audio_from_mic: str = None,
        audio_file: str = None,
        text_input: str = "",
        language: str = "en",
    ):
        if not audio_from_mic and not audio_file and text_input:
            return text_input

        audio_input = audio_from_mic or audio_file

        # 1. Transcribe with original whisper (batched)
        audio = whisperx.load_audio(audio_input)
        model = whisperx.load_model(self.model_name, self.device, device_index=self.device_index, compute_type=self.compute_type)
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)

        # delete model if low on GPU resources
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Align whisper output
        align_model, metadata = whisperx.load_align_model(language_code=result['language'] if language is None else language, device=self.device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, self.device, return_char_alignments=False)

        # delete model if low on GPU resources
        del align_model
        gc.collect()
        torch.cuda.empty_cache()
        
        text_list = [data['text'].strip() + "\n" for data in result["segments"]]
        text = "".join(text_list).rstrip()
        return text
        
    def get_inputs(self):
        return [
            gr.components.Audio(
                label="Record",
                source="microphone",
                type="filepath",
            ),
            gr.components.Audio(
                label="Upload Audio File",
                source="upload",
                type="filepath",
            ),
            gr.components.Textbox(
                lines=2,
                label="Text Input",
            ),
            gr.components.Radio(
                label="Language",
                choices=["en"],
                value="en",
            ),
        ]

    def get_outputs(self):
        return [
            gr.components.Textbox(
                label="Transcript",
            )
        ]
