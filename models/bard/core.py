import os
import gradio as gr
import requests
from bardapi import Bard as BardAPI, SESSION_HEADERS
from utils.model import Model

class Bard(Model):
    def __init__(
        self,
        api_key: str = "",
    ):
        super().__init__()

        self.api_key = api_key or os.getenv("BARD_API_KEY", "")
        assert (
            self.api_key
        ), "Please specify an --bard_api_key"

        self.session = requests.Session()
        self.session.headers = SESSION_HEADERS
        self.session.cookies.set("__Secure-1PSID", self.api_key)
        self.session.cookies.set("__Secure-1PSIDTS", "")
        # session.cookies.set("__Secure-1PSIDCC", "")

        self.setup_interface(self.prompt, self.get_inputs(), self.get_outputs())

    def prompt(
        self,
        input,
    ):
        chat = BardAPI(
            token=self.api_key,
            session=self.session,
        )
        response = chat.get_answer(input)
        reply = response['content']

        return reply
        
    def get_inputs(self):
        return [
            gr.components.Textbox(
                lines=2,
                label="Input",
                placeholder="Tell me about Bard.",
            ),
        ]
    
    def get_outputs(self):
        return [
            gr.components.Textbox(
                lines=5,
                label="Output",
            ),
        ]
