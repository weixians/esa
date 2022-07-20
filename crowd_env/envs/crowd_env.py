import logging
import os

import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm

from crowd_env.envs.env_util import build_sim_action_space, transform_observation
from crowd_env.envs.utils.human import Human
from crowd_env.envs.utils.info import Danger, Nothing, ReachGoal, Collision, Timeout
from crowd_env.envs.utils.state import ObservableState
from crowd_env.envs.utils.utils import point_to_segment_dist


class CrowdEnv:
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.dynamic_human_num = None
        self.static_human_num = None
        self.total_human_num = None
        # for visualization
        self.states = None

        self.sim_action_space = None
        self.action_space_size = None
        self.multiagent_training = True

    def configure(self, config, robot):
        self.robot = robot
        self.config = config
        self.time_limit = config.getint("env", "time_limit")
        self.time_step = config.getfloat("env", "time_step")
        self.randomize_attributes = config.getboolean("env", "randomize_attributes")
        self.success_reward = config.getfloat("reward", "success_reward")
        self.collision_penalty = config.getfloat("reward", "collision_penalty")
        self.discomfort_dist = config.getfloat("reward", "discomfort_dist")
        self.discomfort_penalty_factor = config.getfloat("reward", "discomfort_penalty_factor")
        self.robot_scan_radius = config.getfloat("robot", "scan_radius")
        if self.config.get("humans", "policy") == "orca":
            self.case_capacity = {
                "train": 0,
                "val": 0,
                "test": 0,
            }
            self.case_size = {
                "train": np.iinfo(np.uint32).max - 2000,
                "val": config.getint("env", "val_size"),
                "test": config.getint("env", "test_size"),
            }
            self.train_val_sim = config.get("sim", "train_val_sim")
            self.test_sim = config.get("sim", "test_sim")
            self.square_width = config.getfloat("sim", "square_width")
            self.circle_radius = config.getfloat("sim", "circle_radius")
            self.dynamic_human_num = config.getint("sim", "human_num")
            self.static_human_num = config.getint("sim", "static_human_num")
            self.total_human_num = self.dynamic_human_num + self.static_human_num
            # action space
            self.sim_action_space = build_sim_action_space(
                self.robot.kinematics,
                speed_nums=config.getint("action_space", "v_num"),
                rotation_nums=config.getint("action_space", "angle_num"),
            )
            self.action_space_size = len(self.sim_action_space)
        else:
            raise NotImplementedError

        self.case_counter = {"train": 0, "test": 0, "val": 0}
        logging.info(
            "total human number: {}, dynamic num: {}, static num: {} ".format(
                self.total_human_num, self.dynamic_human_num, self.static_human_num
            )
        )
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info("Training simulation: {}, test simulation: {}".format(self.train_val_sim, self.test_sim))
        logging.info("Square width: {}, circle radius: {}".format(self.square_width, self.circle_radius))

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == "square_crossing":
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == "circle_crossing":
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == "mixed":
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.dynamic_human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, "humans")
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, "humans")
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if (
                                norm((px - agent.px, py - agent.py))
                                < human.radius + agent.radius + self.discomfort_dist
                            ):
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, "humans")
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, "humans")
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError("Episode is not done yet")
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(
            self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref, self.robot.get_velocity()
        )
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning("Simulation cannot terminate!")
            for i in range(self.dynamic_human_num):
                human = self.humans[i]
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i in range(self.dynamic_human_num):
                human = self.humans[i]
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append(
                [
                    self.robot.get_full_state(),
                    [human.get_full_state() for human in self.humans],
                ]
            )

        del sim
        return self.human_times

    def reset(self, phase="test", test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError("robot has to be set!")
        assert phase in ["train", "val", "test"]
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == "test":
            self.human_times = [0] * self.dynamic_human_num
        else:
            self.human_times = [0] * (self.dynamic_human_num if self.multiagent_training else 1)
        if not self.multiagent_training:
            self.train_val_sim = "circle_crossing"

        if self.config.get("humans", "policy") == "trajnet":
            raise NotImplementedError
        else:
            counter_offset = {
                "train": self.case_capacity["val"] + self.case_capacity["test"],
                "val": 0,
                "test": self.case_capacity["val"],
            }
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                if phase != "train":
                    np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ["train", "val"]:
                    human_num = self.dynamic_human_num if self.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.dynamic_human_num, rule=self.test_sim)
                self.add_static_humans(self.static_human_num, self.humans)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == "test"
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.dynamic_human_num = 3
                    self.humans = [Human(self.config, "humans") for _ in range(self.dynamic_human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            if agent.policy is not None:
                agent.policy.time_step = self.time_step

        self.states = list()

        # get current observation
        # ob = [human.get_observable_state() for human in self.humans]
        # return ob
        scanned_humans, _ = self.get_scanned_humans()
        ob = [human.get_observable_state() for human in scanned_humans]
        ob_human_nums = len(ob)
        # if ob_human_nums == 0:
        #     ob = [self.generate_encouraged_ob()]
        for i in range(self.total_human_num - ob_human_nums):
            ob.append(ObservableState(self.robot.px, self.robot.py, 0, 0, 0))

        return transform_observation(self.robot, ob)

    def generate_encouraged_ob(self):
        human_radius = self.config.getfloat("humans", "radius")
        s = self.robot.get_full_state()
        d = ((s.gx - s.px) ** 2 + (s.gy - s.py) ** 2) ** 0.5 + 1e-8
        d2 = d + self.robot.radius + human_radius + 0.1
        hx = s.gx - (s.gx - s.px) * d2 / d
        hy = s.gy - (s.gy - s.py) * d2 / d

        return ObservableState(hx, hy, 0, 0, human_radius)

    def generate_zero_ob(self):
        human_radius = self.config.getfloat("humans", "radius")
        return ObservableState(0, 0, 0, 0, human_radius)

    def get_scanned_humans(self):
        rpx, rpy = self.robot.px, self.robot.py
        scanned_humans = []
        indexes = []
        for i in range(len(self.humans)):
            human = self.humans[i]
            dist = ((human.px - rpx) ** 2 + (human.py - rpy) ** 2) ** (1 / 2)
            if dist <= self.robot_scan_radius:
                scanned_humans.append(human)
                indexes.append(i)
        return scanned_humans, indexes

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        robot_action = self.sim_action_space[action]
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            # 计算出human的下一步动作
            human_actions.append(human.act(ob))

        # collision detection
        collision, dmin = self.detect_robot_human_collision(robot_action)
        # collision detection between humans
        self.detect_humans_collision()

        # check if reaching the goal
        reward, done, info = self.compute_reward_done(robot_action, collision, dmin)

        scanned_humans, indexes = self.get_scanned_humans()
        if update:
            # store state, action value and attention weights
            self.states.append(
                [
                    self.robot.get_full_state(),
                    [human.get_full_state() for human in self.humans],
                ]
            )

            # update all agents
            self.robot.step(robot_action)
            for i, human_action in enumerate(human_actions):
                # only dynamic humans can move
                if i < self.dynamic_human_num:
                    self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i in range(self.dynamic_human_num):
                human = self.humans[i]
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            # ob = [human.get_observable_state() for human in self.humans]
            ob = [human.get_observable_state() for human in scanned_humans]
        else:
            # ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            ob = [
                human.get_next_observable_state(human_actions[index]) for human, index in zip(scanned_humans, indexes)
            ]

        ob_human_nums = len(ob)
        # if ob_human_nums == 0:
        #     ob = [self.generate_encouraged_ob()]
        for i in range(self.total_human_num - ob_human_nums):
            ob.append(ObservableState(self.robot.px, self.robot.py, 0, 0, 0))

        ob = transform_observation(self.robot, ob)
        return ob, reward, done, {"status": info}

    def detect_robot_human_collision(self, robot_action):
        dmin = float("inf")
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == "holonomic":
                vx = human.vx - robot_action.vx
                vy = human.vy - robot_action.vy
            else:
                vx = human.vx - robot_action.v * np.cos(robot_action.r + self.robot.theta)
                vy = human.vy - robot_action.v * np.sin(robot_action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        return collision, dmin

    def detect_humans_collision(self):
        """
        检测人与人之间是否出现碰撞，但其实对于训练来说没啥用，只是打印一下
        """
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx**2 + dy**2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug("Collision happens between humans in step()")

    def compute_reward_done(self, robot_action, collision, dmin):
        end_position = np.array(self.robot.compute_position(robot_action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()
        return reward, done, info

    def render(self, mode="human", save=False, output_dir="", out_prefix=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt

        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

        os.makedirs(output_dir, exist_ok=True)
        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap("hsv", 10)
        robot_color = "yellow"
        robot_scanner_color = "#8AF5E6"
        goal_color = "red"
        arrow_color = "red"
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == "human":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color="b")
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color="r"))
            plt.show()
        elif mode == "traj":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5.5, 5.5)
            ax.set_ylim(-5.5, 5.5)
            ax.set_xlabel("x(m)", fontsize=16)
            ax.set_ylabel("y(m)", fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [
                [self.states[i][1][j].position for j in range(len(self.humans))] for i in range(len(self.states))
            ]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(
                        robot_positions[k],
                        self.robot.radius,
                        fill=True,
                        color=robot_color,
                    )
                    # robot_scanner = plt.Circle(robot_positions[k], self.robot_scan_radius, fill=False, linestyle='--',
                    #                            color=robot_scanner_color)
                    humans = [
                        plt.Circle(
                            human_positions[k][i],
                            self.humans[i].radius,
                            fill=False,
                            color=cmap(i),
                        )
                        for i in range(len(self.humans))
                    ]
                    ax.add_artist(robot)
                    # ax.add_artist(robot_scanner)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = [robot] + humans
                    times = [
                        plt.text(
                            agents[i].center[0] - x_offset,
                            agents[i].center[1] - y_offset,
                            "{:.1f}".format(global_time) if i < self.dynamic_human_num + 1 else "S",
                            color="black",
                            fontsize=14,
                        )
                        for i in range(self.total_human_num + 1)
                    ]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D(
                        (self.states[k - 1][0].px, self.states[k][0].px),
                        (self.states[k - 1][0].py, self.states[k][0].py),
                        color=robot_color,
                        ls="solid",
                    )
                    human_directions = [
                        plt.Line2D(
                            (self.states[k - 1][1][i].px, self.states[k][1][i].px),
                            (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                            color=cmap(i),
                            ls="solid",
                        )
                        for i in range(self.dynamic_human_num)
                    ]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ["Robot"], fontsize=18)
            if save and out_prefix is not None:
                plt.savefig(
                    os.path.join(
                        output_dir,
                        "{}_{}d_{}s.png".format(out_prefix, self.dynamic_human_num, self.static_human_num),
                    )
                )
            # plt.show()
        elif mode == "video":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel("x(m)", fontsize=16)
            ax.set_ylabel("y(m)", fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D(
                [0],
                [5],
                color=goal_color,
                marker="*",
                linestyle="None",
                markersize=15,
                label="Goal",
            )
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            robot_scanner = plt.Circle(
                robot_positions[0],
                self.robot_scan_radius,
                fill=False,
                linestyle="--",
                color=robot_scanner_color,
            )
            ax.add_artist(robot)
            ax.add_artist(robot_scanner)
            ax.add_artist(goal)
            plt.legend([robot, goal], ["Robot", "Goal"], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [
                plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))
            ]
            # human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
            #                           color='black', fontsize=12) for i in range(len(self.humans))]
            human_numbers = []
            for i in range(len(self.humans)):
                # human_numbers.append(plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset,
                #                               str(i), color='black', fontsize=12))
                if i < self.dynamic_human_num:
                    human_numbers.append(
                        plt.text(
                            humans[i].center[0] - x_offset,
                            humans[i].center[1] - y_offset,
                            str(i),
                            color="black",
                            fontsize=12,
                        )
                    )
                else:
                    human_numbers.append(
                        plt.text(
                            humans[i].center[0] - x_offset,
                            humans[i].center[1] - y_offset,
                            "S",
                            color="blue",
                            fontsize=12,
                        )
                    )
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, "Time: {}".format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            # if self.attention_weights is not None:
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == "unicycle":
                orientation = [
                    (
                        (state[0].px, state[0].py),
                        (
                            state[0].px + radius * np.cos(state[0].theta),
                            state[0].py + radius * np.sin(state[0].theta),
                        ),
                    )
                    for state in self.states
                ]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.dynamic_human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(
                            (
                                (agent_state.px, agent_state.py),
                                (
                                    agent_state.px + radius * np.cos(theta),
                                    agent_state.py + radius * np.sin(theta),
                                ),
                            )
                        )
                    orientations.append(orientation)
            arrows = [
                patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                for orientation in orientations
            ]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                robot_scanner.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [
                        patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color, arrowstyle=arrow_style)
                        for orientation in orientations
                    ]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    # if self.attention_weights is not None:
                    #     human.set_color(str(self.attention_weights[frame_num][i]))
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text("Time: {:.2f}".format(frame_num * self.time_step))

            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if save and out_prefix is not None:
                out_path = os.path.join(
                    output_dir,
                    "{}_{}d_{}s.mp4".format(out_prefix, self.dynamic_human_num, self.static_human_num),
                )
                try:
                    ffmpeg_writer = animation.writers["ffmpeg"]
                except RuntimeError:
                    plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
                    ffmpeg_writer = animation.writers["ffmpeg"]
                writer = ffmpeg_writer(fps=8, metadata=dict(artist="Me"), bitrate=1800)
                anim.save(out_path, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError

    def add_static_humans(self, num, humans):
        # randomly initialize static objects in a square of (width, height)
        width = 4
        height = 8
        for i in range(num):
            human = Human(self.config, "humans")
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * width * 0.5 * sign
                py = (np.random.random() - 0.5) * height
                collide = False
                # 检测是否会和其他human距离过小
                for agent in [self.robot] + humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, px, py, 0, 0, 0)
            humans.append(human)
