import torch
from ignite.handlers import Checkpoint


def load_checkpoint(model, checkpoint_fp):
    to_load = {"model": model}
    checkpoint = torch.load(checkpoint_fp)
    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    return checkpoint