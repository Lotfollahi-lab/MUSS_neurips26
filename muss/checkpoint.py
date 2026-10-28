"""Load released weights and Lightning fine-tuning checkpoints."""

from dataclasses import asdict, fields
from pathlib import Path
import os

import torch


def load_checkpoint_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" in state and "cfg" in state:
        return state
    if "state_dict" in state and "muss" in state:
        return {
            "model": {k.removeprefix("model."): v for k, v in state["state_dict"].items()
                      if k.startswith("model.")},
            "cfg": {"model": state["muss"]["model_config"]},
            "meta": {"num_updates": state["global_step"]},
        }
    raise ValueError("Expected a released MUSS checkpoint or a MUSS Lightning checkpoint")


def model_config_from_checkpoint(state):
    from .models import MUSSConfig

    config = state["cfg"]["model"]
    names = {field.name for field in fields(MUSSConfig)}
    missing = names - config.keys()
    if missing:
        raise ValueError(f"Checkpoint model configuration is incomplete: {sorted(missing)}")
    return MUSSConfig(**{k: v for k, v in config.items() if k in names})


def save_weights(model, path, num_updates=0):
    """Export portable weights without optimizer state or dataset embeddings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "cfg": {"model": asdict(model.cfg)}, "meta": {"num_updates": num_updates}}
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, path)
