import numpy as np
from pfrl import agents

from pf_helper.explorer_builder import get_explorer_by_name
from pf_helper.optimizer_builder import get_optimizer_by_name
from pf_helper.replay_buffer_builder import get_replay_buffer_by_name


def build_dqn_agent(args, q_func):
    agent_name = "dqn"

    agent_config = args.agent_config
    explorer = get_explorer_by_name(
        agent_config.get(agent_name, "explorer"),
        args.config_dir,
        args.env.action_space_size,
    )
    optimizer = get_optimizer_by_name(q_func, agent_config.get(agent_name, "optimizer"), args.config_dir)

    rbuf_name = agent_config.get(agent_name, "replay_buffer") if not args.recurrent else "EpisodicReplayBuffer"
    replay_buffer = get_replay_buffer_by_name(rbuf_name, args.config_dir)

    # Now create an agent that will interact with the environment.
    agent = agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        agent_config.getfloat(agent_name, "gamma"),
        explorer,
        replay_start_size=agent_config.getint(agent_name, "replay_start_size"),
        target_update_interval=agent_config.getint(agent_name, "target_update_interval"),
        update_interval=agent_config.getint(agent_name, "update_interval"),
        phi=lambda x: x.astype(np.float32, copy=False),
        gpu=args.gpu,
        recurrent=args.recurrent,
        minibatch_size=agent_config.getint(agent_name, "batch_size"),
        episodic_update_len=agent_config.getint(agent_name, "episodic_update_len"),
    )
    return agent
