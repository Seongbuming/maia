import os
import sys

import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from .utils.callbacks import Iteratorize, Stream
from .utils.prompter import Prompter

from utils.model import Model

class Alpaca(Model):
    def __init__(
        self,
        device: str = "cpu",
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",
        context: bool = True,
    ):
        self.base_model = base_model or os.environ.get("BASE_MODEL", "")
        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.prompter = Prompter(prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.device = device

        if self.device == "cuda":
            self.model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif self.device == "mps":
            self.model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                device_map={"": self.device},
            )
        
        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0 # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()
        
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        
        self.role = "assistant"
        self.messages = []
        self.context = context

        self.setup_interface(self.evaluate, self.get_inputs(), self.get_outputs())
        
    def evaluate(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        if self.context:
            self.messages.append(("user", instruction))
            context = "\n".join([": ".join(message) for message in self.messages])
            prompt = self.prompter.generate_prompt(f"{context}\n{self.role}: ", input)
        else:
            prompt = self.prompter.generate_prompt(instruction, input)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    self.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = self.tokenizer.decode(output)

                    if output[-1] in [self.tokenizer.eos_token_id]:
                        break

                    yield self.prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        response = self.prompter.get_response(output)

        if self.context:
            self.messages.append((self.role, "".join(response)))

        yield response
        
    def get_inputs(self):
        return [
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ]
    
    def get_outputs(self):
        return [
            gr.components.Textbox(
                lines=5,
                label="Response",
            )
        ]
