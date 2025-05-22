import torch
from easydict import EasyDict
from ..task.task import LMTask


def convert_from_sequence_models(task: LMTask, pretrained_path):
    # for mid training
    ckpt = torch.load(pretrained_path, map_location="cuda")
    state_dict = ckpt['task']
    new_state_dict = {}
    for k, v in state_dict.items():
        if "_orig_mod." in k:
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    task.load_state_dict(new_state_dict)


def convert(task, pretrained_path):
    torch.serialization.add_safe_globals([EasyDict])
    if not isinstance(task, LMTask):
        raise NotImplementedError
    if "sequence_model" in pretrained_path:
        convert_from_sequence_models(task, pretrained_path)
    else:
        raise NotImplementedError