import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from utils.model import Model

class Gemma(Model):
    def __init__(
        self,
        api_key: str = "",
        model: str = "google/gemma-7b",
        context: bool = True,
    ):
        super().__init__()

        gemma_token = api_key or os.getenv("GEMMA_TOKEN", "")
        assert (
            gemma_token
        ), "Please specify an --gemma_token"

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=gemma_token)
        self.model = AutoModelForCausalLM.from_pretrained(model, token=gemma_token, device_map='auto')
        self.model = self.accelerator.prepare(self.model)
        self.messages = []
        self.context = context
        
        self.setup_interface(self.prompt, self.get_inputs(), self.get_outputs())

    def prompt(
        self,
        input,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=100,
        history=None,
    ):
        message = f"Human: {input}\n\nAssistant: "
        if type(history) is list:
            self.messages = history + [message]
        elif self.context:
            self.messages.append(message)
        else:
            self.messages = [message]

        input_text = "\n".join(self.messages)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )

        reply = self.tokenizer.decode(output[0], skip_special_tokens=True)
        reply = reply.split("Assistant: ")[-1].strip()

        if self.context:
            self.messages.append(reply)

        return reply

    def get_inputs(self):
        return [
            gr.components.Textbox(
                lines=2,
                label="Input",
                placeholder="Tell me about Gemma.",
            ),
            gr.components.Slider(
                minimum=0, maximum=2, value=0.7, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1, label="Top p"
            ),
            gr.components.Slider(
                minimum=1, maximum=2, value=1, label="Repetition penalty"
            ),
            gr.components.Slider(
                minimum=1, maximum=500, value=100, step=1, label="Max new tokens"
            ),
        ]

    def get_outputs(self):
        return [
            gr.components.Textbox(
                lines=5,
                label="Output",
            ),
        ]
