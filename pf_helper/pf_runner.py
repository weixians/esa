import logging
import os
import time
import torch
import numpy as np

from crowd_env.envs.utils.info import Danger, ReachGoal, Timeout, Collision
from crowd_nav import global_util


def load_pretrained_weights(device, pretrained_model_name, model):
    model_path = os.path.join(global_util.get_project_root(), "data", pretrained_model_name)
    if not os.path.exists(model_path):
        logging.error("model cannot be found by path: {}".format(model_path))
        return
    # model.load_state_dict(torch.load(model_path, device))
    # load pretrained model dicts
    pretrained_model = torch.load(model_path, device)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    logging.info("Pretrained model loaded!")


def train(args, agent):
    phase = "train"
    env = args.env
    writer = args.writer
    train_config = args.train_config
    n_episodes = train_config.getint("train", "train_episodes")

    if args.resume:
        logging.info("Train Resumes.")
        agent.load(args.model_dir)

    val_step = 0
    old_val_success_rate = -np.Inf

    success_times = []
    success_num = 0
    collision_num = 0
    timeout_num = 0
    for i in range(1, n_episodes + 1):
        obs = env.reset(phase=phase)
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            agent.observe(obs, reward, done, False)
            if done:
                if isinstance(info["status"], ReachGoal):
                    success_num += 1
                    writer.add_scalar("train/success", 1, i)
                    writer.add_scalar("train/timeout", 0, i)
                    writer.add_scalar("train/collision", 0, i)
                    success_times.append(env.global_time)
                elif isinstance(info["status"], Timeout):
                    timeout_num += 1
                    writer.add_scalar("train/success", 0, i)
                    writer.add_scalar("train/timeout", 1, i)
                    writer.add_scalar("train/collision", 0, i)
                elif isinstance(info["status"], Collision):
                    collision_num += 1
                    writer.add_scalar("train/success", 0, i)
                    writer.add_scalar("train/timeout", 0, i)
                    writer.add_scalar("train/collision", 1, i)
                break

        statistics = agent.get_statistics()
        writer.add_scalar("{}/reward".format(phase), R, i)
        writer.add_scalar("{}/average_q".format(phase), statistics[0][1], i)
        writer.add_scalar("{}/average_loss".format(phase), statistics[1][1], i)
        writer.add_scalar("{}/nav_time".format(phase), env.global_time, i)

        if i % 10 == 0:
            logging.info("episode:{}, R:{}, info:{}".format(i, R, str(info["status"])))
        if i > 0 and i % 1000 == 0:
            val_step, val_success_rate, _ = eval_agent(args, agent, "val", val_step)
            if val_success_rate + 0.02 >= old_val_success_rate:
                logging.info("正确率提升，保存模型，新正确率：{}，旧正确率：{}".format(val_success_rate, old_val_success_rate))
                old_val_success_rate = val_success_rate
                agent.save(args.model_dir)
            agent.save("{}/epi_{}".format(args.model_dir, i))

    logging.info("--------------训练结束.-------------------")


def eval_agent(args, agent, phase, start_i=0, n_episodes=None):
    env = args.env
    writer = args.writer if hasattr(args, "writer") else None
    env_config = args.env_config
    n_episodes = env_config.getint("env", "{}_size".format(phase)) if n_episodes is None else n_episodes

    logging.info("-------------------Policy: {}, phase: {}， 评估开始-------------------------".format(args.network, phase))

    with agent.eval_mode():
        success_navigation_times = []
        success_running_times = []
        uncomfortable_rates = []
        danger_dist = []
        success_rewards = []
        rewards = []
        successes = []
        inference_times = []

        success_num = 0
        collision_num = 0
        timeout_num = 0

        for i in range(start_i, start_i + n_episodes):
            time_start = time.time()
            obs = env.reset(phase=phase)
            R = 0
            episode_danger_num = 0
            episode_danger_dist = []
            episode_inference_times = []

            while True:
                # Uncomment to watch the behavior in a GUI window
                # env.render()
                infer_start = time.time()
                action = agent.act(obs)
                episode_inference_times.append(time.time() - infer_start)

                obs, r, done, info = env.step(action)
                R += r
                agent.observe(obs, r, done, False)

                if isinstance(info["status"], Danger):
                    episode_danger_num += 1
                    episode_danger_dist.append(info["status"].min_dist)

                if done:
                    inference_times.append(np.average(episode_inference_times))
                    rewards.append(R)
                    if isinstance(info["status"], ReachGoal):
                        success_num += 1
                        success_running_times.append(time.time() - time_start)
                        success_navigation_times.append(env.global_time)
                        uncomfortable_rates.append(episode_danger_num / (env.global_time / env.time_step))
                        danger_dist.append(episode_danger_dist)
                        success_rewards.append(R)
                        successes.append(1)
                    else:
                        # 没成功的episode则填上-1
                        success_running_times.append(-1)
                        success_navigation_times.append(-1)
                        uncomfortable_rates.append(-1)
                        danger_dist.append([])
                        success_rewards.append(-1)
                        successes.append(0)

                        if isinstance(info["status"], Timeout):
                            timeout_num += 1
                        elif isinstance(info["status"], Collision):
                            collision_num += 1
                    break

            if writer is not None:
                writer.add_scalar("{}/reward".format(phase), R, i)
                writer.add_scalar("{}/nav_time".format(phase), env.global_time, i)

    success_rate = success_num / n_episodes
    collision_rate = collision_num / n_episodes
    timeout_rate = timeout_num / n_episodes
    avg_nav_time = np.average(success_navigation_times) if success_navigation_times else env.time_limit

    logging.info("-------------------Policy: {}, phase: {}， 评估结束-------------------------".format(args.network, phase))
    _log_infos(
        start_i + n_episodes,
        writer,
        phase,
        success_rate,
        collision_rate,
        timeout_rate,
        -1,
        avg_nav_time,
    )

    # return start_i + n_episodes, success_num / n_episodes, avg_nav_time
    return (
        # 下一次开始的索引（用于tensorboard打印）
        start_i + n_episodes,
        success_rate,
        (
            success_rate,
            collision_rate,
            timeout_rate,
            success_running_times,
            success_navigation_times,
            uncomfortable_rates,
            success_rewards,
            rewards,
            successes,
            inference_times,
        ),
    )


def visualize_test(args, agent, i):
    env = args.env
    robot = env.robot

    obs = env.reset(phase="test")
    info = ""
    done = False
    last_pos = np.array(robot.get_position())
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        agent.observe(obs, r, done, False)
        current_pos = np.array(robot.get_position())
        logging.debug("Speed: %.2f", np.linalg.norm(current_pos - last_pos) / robot.time_step)
        last_pos = current_pos
    if args.render_mode is not None:
        env.render(
            args.render_mode,
            args.save,
            os.path.join(args.output_render, "{}".format(i)),
            "{}".format(args.network),
        )

    logging.info("It takes %.2f seconds to finish. Final status is %s", env.global_time, info)
    if robot.visible and info == "reach goal":
        human_times = env.get_human_times()
        logging.info(
            "Average time for humans to reach goal: %.2f",
            sum(human_times) / len(human_times),
        )


def _log_infos(
    i,
    writer,
    phase,
    success_rate,
    collision_rate,
    timeout_rate,
    success_uncomfortable_rate,
    avg_nav_time,
):
    if writer is not None:
        writer.add_scalar("{}/success_rate".format(phase), success_rate, i)
        writer.add_scalar("{}/collision_rate".format(phase), collision_rate, i)
        writer.add_scalar("{}/timeout_rate".format(phase), timeout_rate, i)
        writer.add_scalar("{}/success_uncomfortable_rate".format(phase), success_uncomfortable_rate, i)
        writer.add_scalar("{}/avg_nav_time".format(phase), avg_nav_time, i)

    logging.info(
        "success rate: {}, collision rate: {}, timeout_rate: {}, uncomfortable rate: {}, avg_nav_time: {}".format(
            success_rate,
            collision_rate,
            timeout_rate,
            success_uncomfortable_rate,
            avg_nav_time,
        )
    )
