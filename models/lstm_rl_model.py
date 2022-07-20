import torch
import torch.nn as nn
import numpy as np

import pfrl
from crowd_nav.policy.cadrl import mlp


class LstmRlModel(nn.Module):
    def __init__(self, config, input_dim=12, self_state_dim=5, n_actions=0, device=None):
        super().__init__()
        mlp_dims = [int(x) for x in config.get("lstm_rl2", "mlp_dims").split(", ")]
        self.lstm_hidden_dim = config.getint("lstm_rl2", "hn_state_dim")
        self.self_state_dim = self_state_dim
        self.lstm = nn.LSTM(input_dim - self_state_dim, self.lstm_hidden_dim, batch_first=True)
        self.mlp = mlp(self_state_dim + self.lstm_hidden_dim, mlp_dims, last_relu=True)
        self.mlp_values = mlp(mlp_dims[-1], [n_actions])
        self.device = device

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, : self.self_state_dim]

        soted_state = self.sort_states(state)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(soted_state[:, :, self.self_state_dim :], (h0, c0))
        hn = hn.squeeze(0)

        joint_state = torch.cat([self_state, hn], dim=1)
        out = self.mlp(joint_state)
        values = self.mlp_values(out)
        return pfrl.action_value.DiscreteActionValue(values)

    def sort_states(self, states_batch):
        sorted_batch = []
        for batch in states_batch:
            b = batch.numpy()
            sb = sorted(b, key=self.dist, reverse=True)
            sorted_batch.append(sb)
        return torch.from_numpy(np.array(sorted_batch))

    def dist(self, state):
        if np.all(state[5:7]) == 0:
            return np.Inf

        # sort human order by decreasing distance to the robot
        current_dist = (state[5] ** 2 + state[6] ** 2) ** 0.5
        return current_dist
