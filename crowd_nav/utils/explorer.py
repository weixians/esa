import logging
import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from crowd_env.envs.utils.info import *


class Explorer(object):
    def __init__(
        self,
        env,
        robot,
        device,
        memory=None,
        gamma=None,
        target_policy=None,
        writer: SummaryWriter = None,
    ):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.writer = writer
        self.global_count = {}

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(
        self,
        k,
        phase,
        update_memory=False,
        imitation_learning=False,
        episode=None,
        print_failure=False,
        policy_name="",
    ):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success_running_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        for i in tqdm(range(k)):
            if phase not in self.global_count:
                self.global_count[phase] = 1
            else:
                self.global_count[phase] = self.global_count[phase] + 1
            ob = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            time_start = time.time()
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success_running_times.append(time.time() - time_start)
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError("Invalid end signal from environment")

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            c_reward = sum(
                [
                    pow(self.gamma, t * self.robot.time_step * self.robot.v_pref) * reward
                    for t, reward in enumerate(rewards)
                ]
            )
            cumulative_rewards.append(c_reward)

            if self.writer is not None and not imitation_learning:
                self.writer.add_scalar(
                    "{}/cumulative_reward".format(phase),
                    c_reward,
                    self.global_count[phase],
                )

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        if self.writer is not None and not imitation_learning:
            self.writer.add_scalar("{}/success_rate".format(phase), success_rate, self.global_count[phase])
            self.writer.add_scalar(
                "{}/collision_rate".format(phase),
                collision_rate,
                self.global_count[phase],
            )
            self.writer.add_scalar("{}/avg_nav_time".format(phase), avg_nav_time, self.global_count[phase])

        extra_info = "" if episode is None else "in train_latest {} ".format(episode)
        logging.info(
            "{} {:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}".format(
                policy_name,
                phase.upper(),
                extra_info,
                success_rate,
                collision_rate,
                avg_nav_time,
                average(cumulative_rewards),
            )
        )

        num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        # if phase in ['val', 'test']:
        logging.info(
            "Frequency of being in danger: %.2f and average min separate distance in danger: %.2f",
            too_close / num_step,
            average(min_dist),
        )

        if print_failure:
            logging.info("Collision cases: " + " ".join([str(x) for x in collision_cases]))
            logging.info("Timeout cases: " + " ".join([str(x) for x in timeout_cases]))

        success_running_time = np.average(success_running_times) if success_running_times else self.env.time_limit
        return (
            success_rate,
            average(cumulative_rewards),
            [
                success_rate,
                collision_rate,
                timeout_rate,
                # 危险比例
                too_close / num_step,
                # 平均导航时间（时间步，不计入模型预测所花时间）
                avg_nav_time,
                # 模型运行一个episode的平均所花时间
                success_running_time,
                average(cumulative_rewards),
            ],
        )

    def update_memory(self, states, actions, rewards, imitation_learning=False, num_recent_states=3):
        if self.memory is None or self.gamma is None:
            raise ValueError("Memory or gamma value is not set!")

        recent_states = []
        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum(
                    [
                        pow(
                            self.gamma,
                            max(t - i, 0) * self.robot.time_step * self.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= i else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            if len(recent_states) >= num_recent_states:
                recent_states.pop(0)
            recent_states.append(state.cpu().numpy()[0, :6])
            if i == 0:
                for _ in range(num_recent_states - 1):
                    recent_states.append(state.cpu().numpy()[0, :6])

            self.memory.push((state.cpu().numpy(), value.cpu().numpy(), copy.deepcopy(recent_states)))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
