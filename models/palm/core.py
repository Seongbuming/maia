import os
import gradio as gr
import google.generativeai as palm
from utils.model import Model

class PaLM(Model):
    def __init__(
        self,
        api_key: str = "",
        model: str = "models/chat-bison-001",
        context: bool = True,
    ):
        super().__init__()
        
        self.api_key = api_key or os.getenv("PALM_API_KEY", "")
        assert (
            self.api_key
        ), "Please specify an --palm_api_key"
        palm.configure(api_key=self.api_key)

        self.model_name = model
        self.model = palm.get_model(model)
        self.messages = []
        self.context = context

        self.setup_interface(self.prompt, self.get_inputs(), self.get_outputs())

    def prompt(
        self,
        input,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        stop=[],
        history=None,
    ):
        message = {
            "author": "user",
            "content": input,
        }

        if type(history) is list:
            self.messages = history + [message]
        elif self.context:
            self.messages.append(message)
        else:
            self.messages = [message]

        if self.model_name == "models/chat-bison-001":
            chat = palm.chat(
                model=self.model,
                messages=self.messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            print(chat)
            author = chat.messages[-1]['author']
            reply = chat.messages[-1]['content']
        else:
            author = "assistant"
            reply = palm.generate_text(
                model=self.model,
                prompt=input,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop,
            )
            reply = reply.result

        if self.context:
            self.messages.append({
                "author": author,
                "content": reply
            })

        return reply
    
    def get_inputs(self):
        return [
            gr.components.Textbox(
                lines=2,
                label="Input",
                placeholder="Tell me about PaLM.",
            ),
            gr.components.Slider(
                minimum=0, maximum=2, value=0.25, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.95, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
        ]
    
    def get_outputs(self):
        return [
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ]
