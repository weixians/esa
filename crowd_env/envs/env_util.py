import itertools
import numpy as np
import torch

from crowd_env.envs.utils.action import ActionXY, ActionRot


def build_sim_action_space(kinematics, v_pref=1.0, speed_nums=4, rotation_nums=8):
    """
    Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
    """
    holonomic = True if kinematics == "holonomic" else False
    # 动作空间共有5种不同速度，16种不同旋转角度
    speeds = [(np.exp((i + 1) / speed_nums) - 1) / (np.e - 1) * v_pref for i in range(speed_nums)]
    if holonomic:
        # 线性平分360度
        rotations = np.linspace(0, 2 * np.pi, rotation_nums, endpoint=False)
    else:
        # 线性平分-45到+45度
        rotations = np.linspace(-np.pi / 4, np.pi / 4, rotation_nums)

    action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        if holonomic:
            action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
        else:
            action_space.append(ActionRot(speed, rotation))

    return action_space


def transform_observation(robot, human_states):
    """
    将observation转化为符合gym.env允许的格式
    """

    state_tensor = torch.cat(
        [torch.Tensor([robot.get_full_state() + human_state]) for human_state in human_states],
        dim=0,
    )
    state = transform_coordinates(state_tensor)
    return state


def transform_coordinates(state):
    """
    Transform the coordinate to agent-centric.
    将robot 及 human 的坐标修改为以 robot 为中心的坐标
    Input state tensor is of size (batch_size, state_length)

    """
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    # robot与goal之间的距离
    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    # human的state信息
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    # 将robot与human各自的半径相加，用于之后的检测碰撞
    radius_sum = radius + radius1
    # robot与neighbor i 之间的距离
    du = torch.norm(
        torch.cat(
            [
                (state[:, 0] - state[:, 9]).reshape((batch, -1)),
                (state[:, 1] - state[:, 10]).reshape((batch, -1)),
            ],
            dim=1,
        ),
        2,
        dim=1,
        keepdim=True,
    )
    new_state = torch.cat([dg, v_pref, radius, vx, vy, px1, py1, vx1, vy1, radius1, du, radius_sum], dim=1)
    # new_state = torch.cat([dg, vx, vy, px1, py1, vx1, vy1, du], dim=1)
    return new_state.detach().cpu().numpy()
