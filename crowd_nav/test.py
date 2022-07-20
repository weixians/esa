import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_env.envs.utils.robot import Robot
from crowd_env.envs.policy.orca import ORCA


def main_test(
    policy,
    out_dir_suffix="",
    model_subdir=None,
    model_suffix="",
    dynamic_human_num=5,
    static_human_num=0,
    time_limit=25,
    visualize=False,
    test_case=0,
    traj=False,
    save_video=False,
    video_dir="out_video",
    test_size=None,
    policy_name="",
):
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--env_config", type=str, default="env.config")
    parser.add_argument("--policy_config", type=str, default="policy.config")
    parser.add_argument("--policy", type=str, default=policy)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--il", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--visualize", default=visualize, action="store_true")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--test_case", type=int, default=test_case)
    parser.add_argument("--square", default=False, action="store_true")
    parser.add_argument("--circle", default=False, action="store_true")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--traj", default=traj, action="store_true")
    args = parser.parse_args()

    if args.visualize and args.video_file is None and save_video:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        args.video_file = os.path.join(video_dir, policy + "-" + out_dir_suffix)

    if out_dir_suffix != "":
        args.model_dir = os.path.join(args.out_dir, args.policy + "-" + out_dir_suffix)
    else:
        args.model_dir = os.path.join(args.out_dir, args.policy)
    env_config_file = os.path.join(args.model_dir, args.env_config)
    policy_config_file = os.path.join(args.model_dir, args.policy_config)
    if model_subdir is not None:
        args.model_dir = os.path.join(args.model_dir, model_subdir)
    if args.model_dir is not None:
        if args.il:
            model_weights = os.path.join(args.model_dir, "il_model.pth")
        else:
            if os.path.exists(os.path.join(args.model_dir, "resumed_rl_model.pth")):
                model_weights = os.path.join(args.model_dir, "resumed_rl_model.pth")
            else:
                model_weights = os.path.join(args.model_dir, "rl_model.pth" + model_suffix)

    # configure logging and device
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error("Trainable policy must be specified with a model weights directory")
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    # 设置人数
    env_config.set("sim", "human_num", str(dynamic_human_num))
    env_config.set("sim", "static_human_num", str(static_human_num))
    env_config.set("env", "time_limit", str(time_limit))
    if test_size is not None:
        env_config.set("env", "test_size", str(test_size))
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)
    if args.square:
        env.test_sim = "square_crossing"
    if args.circle:
        env.test_sim = "circle_crossing"

    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9, writer=None)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info("ORCA agent buffer: %f", robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug("Speed: %.2f", np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render("traj", args.video_file)
        else:
            env.render("video", args.video_file)

        logging.info("It takes %.2f seconds to finish. Final status is %s", env.global_time, info)
        if robot.visible and info == "reach goal":
            human_times = env.get_human_times()
            logging.info(
                "Average time for humans to reach goal: %.2f",
                sum(human_times) / len(human_times),
            )
    else:
        return explorer.run_k_episodes(
            env.case_size[args.phase],
            args.phase,
            print_failure=True,
            policy_name="{},dynamic:{},static:{}; ".format(policy_name, dynamic_human_num, static_human_num),
        )
