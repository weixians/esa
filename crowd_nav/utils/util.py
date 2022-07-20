import torch


def rotate(state):
    """
    Transform the coordinate to agent-centric.
    Input state tensor is of size (batch_size, state_length)

    """
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    # 计算到goal的距离
    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    # set theta to be zero since it's not used
    theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(
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
    new_state = torch.cat(
        [
            dg,
            v_pref,
            theta,
            radius,
            vx,
            vy,
            px1,
            py1,
            vx1,
            vy1,
            radius1,
            da,
            radius_sum,
        ],
        dim=1,
    )
    return new_state
