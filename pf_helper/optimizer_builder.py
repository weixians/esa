import os
from configparser import RawConfigParser

import torch


def get_optimizer_by_name(model, name, config_dir):
    config = RawConfigParser()
    config.read(os.path.join(config_dir, "optimizer.cfg"))

    if name == "adam":
        return build_adam_optimizer(model, config)

    raise ValueError("Cannot find optimizer by name {}".format(name))


def build_adam_optimizer(model, config: RawConfigParser):
    opt = torch.optim.Adam(model.parameters(), config.getfloat("adam", "lr"), eps=config.getfloat("adam", "eps"))
    return opt
