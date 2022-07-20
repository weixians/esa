import os
from configparser import RawConfigParser

import numpy as np
from pfrl import explorers


def get_explorer_by_name(name, config_dir, n_actions=None):
    config = RawConfigParser()
    config.read(os.path.join(config_dir, "explorer.cfg"))

    if name == "greedy":
        return explorers.Greedy()
    elif name == "LinearDecayEpsilonGreedy":
        return explorers.LinearDecayEpsilonGreedy(
            start_epsilon=config.getfloat(name, "start_epsilon"),
            end_epsilon=config.getfloat(name, "end_epsilon"),
            decay_steps=config.getfloat(name, "decay_steps"),
            random_action_func=lambda: np.random.randint(n_actions),
        )
    raise ValueError("Cannot find optimizer by name {}".format(name))
