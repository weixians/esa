# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import torch
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory2 import ReplayBuffer
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def run(args, only_imitation_learn=False):
    train_config = args.train_config
    robot = args.env.robot

    model = args.policy.get_model()
    memory = ReplayBuffer(train_config.getint("train", "capacity"), args.device)
    trainer = Trainer(
        model,
        memory,
        args.device,
        train_config.getint("trainer", "batch_size"),
        writer=args.writer,
    )
    explorer = Explorer(
        args.env,
        args.env.robot,
        args.device,
        memory,
        args.policy.gamma,
        target_policy=args.policy,
        writer=args.writer,
    )

    if only_imitation_learn:
        imitation_learning(args, train_config, trainer, robot, explorer, model, memory)
        return

    # load trained rl model for resume training
    if args.resume:
        if not os.path.exists(args.rl_model_path):
            raise FileNotFoundError("RL model is needed for resume training")
        model.load_state_dict(torch.load(args.rl_model_path, args.device))
        logging.info("Load reinforcement learning trained weights. Resume training")
    # load trained il model
    elif os.path.exists(args.il_model_path):
        model.load_state_dict(torch.load(args.il_model_path))
        logging.info("Load imitation learning trained weights.")
    else:
        imitation_learning(args, train_config, trainer, robot, explorer, model, memory)

    explorer.update_target_model(model)
    # reinforcement learning
    reinforcement_learning(args, train_config, trainer, robot, explorer, model, memory)
    # final test
    explorer.run_k_episodes(args.env.case_size["test"], "test")


def imitation_learning(args, train_config, trainer, robot, explorer, model, memory):
    il_policy = train_config.get("imitation_learning", "il_policy")
    trainer.set_learning_rate(train_config.getfloat("imitation_learning", "il_learning_rate"))
    if robot.visible:
        safety_space = 0
    else:
        safety_space = train_config.getfloat("imitation_learning", "safety_space")
    il_policy = policy_factory[il_policy]()
    il_policy.multiagent_training = args.policy.multiagent_training
    il_policy.safety_space = safety_space
    robot.set_policy(il_policy)
    explorer.run_k_episodes(
        train_config.getint("imitation_learning", "il_episodes"),
        "train",
        update_memory=True,
        imitation_learning=True,
    )
    trainer.il_optimize_epoch(train_config.getint("imitation_learning", "il_epochs"))
    torch.save(model.state_dict(), args.il_model_path)
    logging.info("Finish imitation learning. Weights saved.")
    logging.info("Experience set size: %d/%d", len(memory), memory.capacity)


def reinforcement_learning(args, train_config, trainer, robot, explorer, model, memory):
    epsilon_start = train_config.getfloat("train", "epsilon_start")
    epsilon_end = train_config.getfloat("train", "epsilon_end")
    epsilon_decay = train_config.getfloat("train", "epsilon_decay")

    args.policy.set_env(args.env)
    robot.set_policy(args.policy)
    robot.print_info()
    trainer.set_learning_rate(train_config.getfloat("train", "rl_learning_rate"))
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, "train", update_memory=True, episode=0)
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    episode = 0
    old_success_rate, old_avg_cu_rewards = -np.inf, -np.inf
    while episode < train_config.getint("train", "train_episodes"):
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model, episode小于3000次时评估没意义
        if episode % train_config.getint("train", "evaluation_interval") == 0:
            # 获取评估的平均reward以及成功率，并与上次对比，决定是否保存
            success_rate, avg_cu_rewards, _ = explorer.run_k_episodes(args.env.case_size["val"], "val", episode=episode)

            if success_rate >= old_success_rate:
                logging.info(
                    "测试正确率上升---保存模型，此次成功率：{}，平均累加rewards：{}；【上次成功率：{}，平均累加rewards：{}】".format(
                        success_rate,
                        avg_cu_rewards,
                        old_success_rate,
                        old_avg_cu_rewards,
                    )
                )
                old_success_rate = success_rate
                old_avg_cu_rewards = avg_cu_rewards
                torch.save(model.state_dict(), args.rl_model_path)

        # sample k episodes into memory and optimize over the generated memory
        success_rate, avg_cu_rewards, _ = explorer.run_k_episodes(
            train_config.getint("train", "sample_episodes"),
            "train",
            update_memory=True,
            episode=episode,
        )
        trainer.rl_optimize_batch(train_config.getint("train", "train_batches"))
        episode += 1

        if episode % train_config.getint("train", "target_update_interval") == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % train_config.getint("train", "checkpoint_interval") == 0:
            # save the latest model
            torch.save(model.state_dict(), os.path.join(args.model_dir, "rl_latest_model.pth"))
            # save the episode model
            os.makedirs(os.path.join(args.model_dir, "episode"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.model_dir, "episode", "rl_model_{}.pth".format(episode)),
            )
