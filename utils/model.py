import os
import gradio as gr
import typing

class Model:
    def __init__(self) -> None:
        self.inputs = []
        self.fn = None
        self.outputs = []
    
    def setup_interface(
        self,
        fn: typing.Callable,
        inputs: typing.List[gr.components.Component],
        outputs: typing.List[gr.components.Component],
    ) -> None:
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs

    def prompt(*args, **kwargs) -> str:
        return

    def setup_ddp(
        self,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        import torch
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    def clean_ddp(self) -> None:
        import torch.distributed as dist
        dist.destroy_process_group()
