# -*- coding: utf-8 -*-
import sys
import logging
import argparse
import configparser
import os
import shutil

import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter

from crowd_env.envs.utils.robot import Robot
from crowd_nav import global_util
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory2 import ReplayBuffer
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def copy_configs_to_output(args):
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.test or args.resume:
            make_new_dir = False
        else:
            key = input("Output directory already exists! Overwrite the folder? (y/n)")
            if key == "y" and not args.resume:
                make_new_dir = True
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        # make other dirs
        os.makedirs(os.path.join(args.output_dir, "train_best"))
        os.makedirs(os.path.join(args.output_dir, "train_latest"))
        os.makedirs(os.path.join(args.output_dir, "configs"))
        # copy configs
        shutil.copytree(args.config_dir, os.path.join(args.output_dir, "configs"))

    config_dir = os.path.join(args.output_dir, "configs")
    args.policy_config = os.path.join(config_dir, "policy.config")
    args.env_config = os.path.join(config_dir, "env.config")
    args.train_config = os.path.join(config_dir, "train.config")


def parse_args():
    pass


def main_train(
    default_policy_name,
    use_gpu=False,
    cuda_index=0,
    resume=False,
):
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--policy", type=str, default=default_policy_name)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(global_util.get_project_root(), "output"),
    )
    parser.add_argument("--weights", type=str)
    parser.add_argument("--resume", default=resume, action="store_true")
    parser.add_argument("--gpu", default=use_gpu, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.policy)

    # configure paths
    copy_configs_to_output(args)

    log_file = os.path.join(args.output_dir, "output.log")
    il_weight_file = os.path.join(args.output_dir, "il_model.pth")
    rl_weight_file = os.path.join(args.output_dir, "rl_model.pth")
    rl_weight_file_train_best = os.path.join(os.path.join(args.output_dir, "train_best"), "rl_model.pth")
    rl_weight_file_train_latest = os.path.join(os.path.join(args.output_dir, "train_latest"), "rl_model.pth")

    # configure logging
    mode = "a" if args.resume else "w"
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda:{}".format(cuda_index) if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    config_dir = os.path.join(args.output_dir, "configs")
    args.policy_config = os.path.join(config_dir, "policy.config")
    args.env_config = os.path.join(config_dir, "env.config")
    args.train_config = os.path.join(config_dir, "train.config")

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error("Policy has to be trainable")
    if args.policy_config is None:
        parser.error("Policy config has to be specified for a trainable network")
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.set_device(device)
    policy.configure(policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)

    # read training parameters
    if args.train_config is None:
        parser.error("Train config has to be specified for a trainable network")
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    rl_learning_rate = train_config.getfloat("train", "rl_learning_rate")
    train_batches = train_config.getint("train", "train_batches")
    train_episodes = train_config.getint("train", "train_episodes")
    sample_episodes = train_config.getint("train", "sample_episodes")
    target_update_interval = train_config.getint("train", "target_update_interval")
    evaluation_interval = train_config.getint("train", "evaluation_interval")
    capacity = train_config.getint("train", "capacity")
    epsilon_start = train_config.getfloat("train", "epsilon_start")
    epsilon_end = train_config.getfloat("train", "epsilon_end")
    epsilon_decay = train_config.getfloat("train", "epsilon_decay")
    checkpoint_interval = train_config.getint("train", "checkpoint_interval")

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    model = policy.get_model()
    batch_size = train_config.getint("trainer", "batch_size")
    # configure trainer and explorer
    robot = Robot(env_config, "robot")
    env.set_robot(robot)
    memory = ReplayBuffer(capacity, args.device)
    trainer = Trainer(model, memory, device, batch_size, writer=writer)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy, writer=writer)

    # imitation learning
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error("RL weights does not exist")
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, "resumed_rl_model.pth")
        logging.info("Load reinforcement learning trained weights. Resume training")
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info("Load imitation learning trained weights.")
    else:
        il_episodes = train_config.getint("imitation_learning", "il_episodes")
        il_policy = train_config.get("imitation_learning", "il_policy")
        il_epochs = train_config.getint("imitation_learning", "il_epochs")
        il_learning_rate = train_config.getfloat("imitation_learning", "il_learning_rate")
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat("imitation_learning", "safety_space")
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, "train", update_memory=True, imitation_learning=True)
        trainer.il_optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info("Finish imitation learning. Weights saved.")
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, "train", update_memory=True, episode=0)
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    episode = 0
    old_success_rate, old_avg_cu_rewards = -np.inf, -np.inf
    success_train_count = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model, episode小于3000次时评估没意义
        if episode % evaluation_interval == 0:
            # 获取评估的平均reward以及成功率，并与上次对比，决定是否保存
            success_rate, avg_cu_rewards, _ = explorer.run_k_episodes(env.case_size["val"], "val", episode=episode)

            if success_rate >= old_success_rate:
                logging.info(
                    "评估保存模型，此次成功率：{}，平均累加rewards：{}；【上次成功率：{}，平均累加rewards：{}】".format(
                        success_rate,
                        avg_cu_rewards,
                        old_success_rate,
                        old_avg_cu_rewards,
                    )
                )
                old_success_rate = success_rate
                old_avg_cu_rewards = avg_cu_rewards
                torch.save(model.state_dict(), rl_weight_file)

        # sample k episodes into memory and optimize over the generated memory
        success_rate, avg_cu_rewards, _ = explorer.run_k_episodes(
            sample_episodes, "train", update_memory=True, episode=episode
        )
        if success_rate == 1:
            success_train_count += 1
        else:
            # 新增，未跑
            success_train_count = 0
        trainer.rl_optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), "{}_{}".format(rl_weight_file_train_latest, episode))
        if success_train_count >= 50:
            torch.save(model.state_dict(), rl_weight_file_train_best)
            success_train_count = 0

    # final test
    explorer.run_k_episodes(env.case_size["test"], "test", episode=episode)


if __name__ == "__main__":
    main_train("sarl")
