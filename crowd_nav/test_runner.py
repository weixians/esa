import logging

import numpy as np

from crowd_env.envs.policy.orca import ORCA
from crowd_nav.utils.explorer import Explorer


def run(args, test_n_episodes=None):
    env = args.env
    robot = env.robot
    policy = args.policy
    args.policy.set_phase("test")

    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info("ORCA agent buffer: %f", robot.policy.safety_space)

    explorer = Explorer(
        env,
        env.robot,
        args.device,
        gamma=policy.gamma,
        writer=None,
    )

    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()

    if args.visualize:
        for i in range(10):
            ob = env.reset("test")
            info = ""
            done = False
            last_pos = np.array(robot.get_position())
            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug(
                    "Speed: %.2f",
                    np.linalg.norm(current_pos - last_pos) / robot.time_step,
                )
                last_pos = current_pos
            if args.render_mode is not None:
                env.render(
                    args.render_mode,
                    args.save,
                    args.output_render,
                    "{}_{}".format(policy.name, i),
                )

            logging.info(
                "It takes %.2f seconds to finish. Final status is %s",
                env.global_time,
                info,
            )
            if robot.visible and info == "reach goal":
                human_times = env.get_human_times()
                logging.info(
                    "Average time for humans to reach goal: %.2f",
                    sum(human_times) / len(human_times),
                )
    else:
        return explorer.run_k_episodes(
            env.case_size["test"] if test_n_episodes is None else test_n_episodes,
            "test",
            print_failure=True,
            policy_name="{},dynamic:{},static:{}; ".format(policy.name, args.dynamic_num, args.static_num),
        )
