from utils.model import Model
from conversation.prompter import BasePrompter, MAIAPrompter

class Pipeline(Model):
    def __init__(
        self,
        transcribe_model: Model,
        generate_model: Model,
        #synthesize_model: Model,
        forced_response: str = "",
    ):
        self.transcribe = transcribe_model.fn
        self.generate = generate_model.fn
        #self.synthesize = synthesize_model.fn
        self.forced_response = forced_response

        self.setup_interface(
            fn=self,
            inputs=transcribe_model.inputs,
            outputs=[
                *transcribe_model.outputs,
                *generate_model.outputs,
                #*synthesize_model.outputs,
            ],
        )
    
    def __call__(self, *args, **kwargs):
        transcript = self.transcribe(*args, **kwargs)

        if self.forced_response:
            message = self.forced_response
        else:
            message = "".join(self.generate(transcript))

        #speech = self.synthesize(message)

        return transcript, message

class PairwisePipeline(Pipeline):
    def __init__(
        self,
        transcribe_model: Model,
        generate_model_1: BasePrompter,
        generate_model_2: MAIAPrompter,
        forced_response: str = "",
    ):
        self.transcribe_model = transcribe_model
        self.generate_model_1 = generate_model_1
        self.generate_model_2 = generate_model_2

        self.transcribe = transcribe_model.fn
        self.generate_1 = generate_model_1.fn
        self.generate_2 = generate_model_2.fn

        self.forced_response = forced_response

        self.setup_interface(
            fn=self,
            inputs=transcribe_model.inputs,
            outputs=[
                *transcribe_model.outputs,
                *generate_model_1.outputs,
                *generate_model_2.outputs,
            ],
        )
    
    def __call__(self, *args, **kwargs):
        transcript = self.transcribe(*args, **kwargs)

        if self.forced_response:
            message1 = message2 = self.forced_response
        else:
            if not self.generate_model_1.model_loaded:
                self.generate_model_1.reset()
            if not self.generate_model_2.model_loaded:
                self.generate_model_2.reset()

            message1 = "".join(self.generate_1(transcript))
            message2 = "".join(self.generate_2(transcript))

        return transcript, message1, message2
