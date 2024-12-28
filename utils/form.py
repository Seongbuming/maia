import gradio as gr
from .pipeline import Pipeline, PairwisePipeline

class Form:
    def __init__(
        self,
        model: Pipeline,
        title: str,
    ):
        self.model = model
        self.title = title
        self._form = self._create_form()
    
    def _create_form(self) -> gr.Blocks:
        with gr.Blocks() as form:
            gr.Interface(
                fn=self.model.fn,
                inputs=self.model.inputs,
                outputs=self.model.outputs,
                title=self.title,
            )
        
        return form

    def get_form(self) -> gr.Blocks:
        if self._form is None:
            raise NotImplementedError("The _form attribute has not been initialized.")
        return self._form

class PairwiseForm(Form):
    def __init__(
        self,
        model: PairwisePipeline,
        title: str,
    ):
        self.model = model
        self.title = title
        self._form = self._create_form()
    
    def _create_form(self) -> gr.Blocks:
        with gr.Blocks() as form:
            gr.Interface(
                fn=self.model.fn,
                inputs=self.model.inputs,
                outputs=self.model.outputs,
                title=self.title,
            )
        
        return form

    def get_form(self) -> gr.Blocks:
        if self._form is None:
            raise NotImplementedError("The _form attribute has not been initialized.")
        return self._form
