import os
from configparser import RawConfigParser

from pfrl import replay_buffers


def get_replay_buffer_by_name(name, config_dir):
    config = RawConfigParser()
    config.read(os.path.join(config_dir, "replay_buffer.cfg"))

    if name == "PrioritizedReplayBuffer":
        return replay_buffers.PrioritizedReplayBuffer(
            capacity=config.getint(name, "capacity"),
            alpha=config.getfloat(name, "alpha"),
            beta0=config.getfloat(name, "beta0"),
            betasteps=config.getint(name, "betasteps"),
            num_steps=config.getint(name, "num_steps"),
            normalize_by_max=config.get(name, "normalize_by_max"),
        )
    elif name == "EpisodicReplayBuffer":
        return replay_buffers.EpisodicReplayBuffer(capacity=config.getint(name, "capacity"))
    elif name == "PrioritizedEpisodicReplayBuffer":
        return replay_buffers.PrioritizedEpisodicReplayBuffer(
            capacity=config.getint(name, "capacity"),
            alpha=config.getfloat(name, "alpha"),
            beta0=config.getfloat(name, "beta0"),
            betasteps=config.getint(name, "betasteps"),
        )

    raise ValueError("Cannot find optimizer by name {}".format(name))
