# -*- coding: utf-8 -*-
import argparse
import configparser
import os
import shutil
import torch
from tensorboardX import SummaryWriter

from crowd_env.envs import CrowdSim
from crowd_env.envs.crowd_env import CrowdEnv
from crowd_env.envs.utils.robot import Robot
from crowd_nav import global_util
from crowd_nav.policy.policy_factory import policy_factory


def initialize(
    default_policy_name,
    default_random_seed=0,
    default_dynamic_num=5,
    default_static_num=0,
    test_time_limit=25,
    default_output_dir=os.path.join(global_util.get_project_root(), "../output_esa"),
    default_test=False,
    use_pfrl=False,
    logger=None,
    test_model_name=None,
):
    args = parse_args(
        default_policy_name,
        default_random_seed,
        default_dynamic_num,
        default_static_num,
        default_output_dir,
        default_test,
    )
    args.use_pfrl = use_pfrl
    # set random seeds
    global_util.set_random_seeds(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.policy)
    args.model_dir = os.path.join(args.output_dir, args.model_dir)
    load_config_path(args)
    if logger is not None:
        args.logger = logger
    else:
        init_logger(args)
    load_device(args)
    config_env(args, use_pfrl, test_time_limit)
    config_policy(args, test_model_name)
    config_train(args)
    load_other_configs(args)
    return args


def parse_args(
    default_policy_name,
    default_random_seed,
    default_dynamic_num=5,
    default_static_num=0,
    default_output_dir="",
    default_test=False,
):
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--seed", type=int, default=default_random_seed, help="seed of the random")
    parser.add_argument("--policy", type=str, default=default_policy_name)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.join(global_util.get_project_root(), "configs"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model",
        help="folder path to save/load neural network models",
    )
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument(
        "--load_pretrain",
        default=False,
        action="store_true",
        help="Whether to load pretrained model",
    )

    # --------- used only in test -----------
    parser.add_argument("--test", default=default_test, action="store_true")
    parser.add_argument(
        "--dynamic_num",
        type=int,
        default=default_dynamic_num,
        help="dynamic human number",
    )
    parser.add_argument(
        "--static_num",
        type=int,
        default=default_static_num,
        help="static obstacle number",
    )
    parser.add_argument("--test_case", type=int, default=0)
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="render mode: human, traj or video",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="whether to save the rendered",
    )
    parser.add_argument(
        "--use_latest",
        default=False,
        action="store_true",
        help="Whether to use the latest trained model. If not, use the best one",
    )
    # --------- used only in test -----------

    args = parser.parse_args()
    return args


def load_config_path(args):
    make_new_dir = True
    if args.test or args.resume:
        make_new_dir = False
        if not os.path.exists(args.output_dir):
            raise FileNotFoundError("cannot find directory by path: {}".format(args.output_dir))
        if args.save:
            args.output_render = os.path.join(args.output_dir, "render")
            os.makedirs(args.output_render, exist_ok=True)

    elif os.path.exists(args.output_dir):
        key = input("Output directory already exists! Overwrite the folder? (y/n)")
        if key == "y" and not args.resume:
            make_new_dir = True
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False

    if make_new_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        # copy configs
        shutil.copytree(args.config_dir, os.path.join(args.output_dir, "configs"))
    config_dir = os.path.join(args.output_dir, "configs")
    args.config_dir = config_dir
    args.policy_config = os.path.join(config_dir, "policy.config")
    args.env_config = os.path.join(config_dir, "env.config")
    args.train_config = os.path.join(config_dir, "train.config")
    args.agent_config = os.path.join(config_dir, "agent.cfg")


def load_device(args):
    if torch.cuda.is_available():
        args.logger.info("CUDA is available")
        if args.gpu >= 0:
            args.device = torch.device("cuda:{}".format(args.gpu))
        else:
            args.device = torch.device("cpu")
        args.logger.info("Use device: {}".format(args.device))
    else:
        args.device = torch.device("cpu")
        args.logger.info("CUDA is not available")
        args.gpu = -1


def init_logger(args):
    if args.resume:
        log_name = "resume.log"
    elif args.test:
        log_name = "test.log"
    else:
        log_name = "train.log"
    args.logger = global_util.setup_logger(os.path.join(args.output_dir, log_name), use_file_log=True)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))


def config_env(args, use_pfrl, test_time_limit):
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    if args.test:
        env_config.set("sim", "human_num", str(args.dynamic_num))
        env_config.set("sim", "static_human_num", str(args.static_num))
        env_config.set("env", "time_limit", str(test_time_limit))

    env = CrowdEnv() if use_pfrl else CrowdSim()
    robot = Robot(env_config, "robot")
    env.configure(env_config, robot)
    args.env = env
    args.env_config = env_config


def config_policy(args, test_model_name):
    # configure policy
    args.network = args.policy
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    args.policy_config = policy_config
    if args.use_pfrl:
        return

    policy = policy_factory[args.policy]()
    policy.set_device(args.device)
    policy.configure(policy_config)
    args.policy = policy
    if not policy.trainable:
        raise ValueError("Policy has to be trainable")

    args.il_model_path = os.path.join(args.model_dir, "il_model.pth")
    args.rl_model_path = os.path.join(args.model_dir, "rl_model.pth")
    # load trained model when testing
    if args.test:
        if args.use_latest:
            args.rl_model_path = os.path.join(args.model_dir, "rl_latest_model.pth")
        elif test_model_name is not None:
            args.rl_model_path = os.path.join(args.model_dir, "episode", test_model_name)
        policy.get_model().load_state_dict(torch.load(args.rl_model_path, args.device))


def config_train(args):
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    args.train_config = train_config


def load_other_configs(args):
    agent_config = configparser.RawConfigParser()
    agent_config.read(args.agent_config)
    args.agent_config = agent_config
