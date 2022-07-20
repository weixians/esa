import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

import pfrl
from crowd_nav.policy.cadrl import mlp


class EsaModel(nn.Module):
    """
    LSTM部分只编码人，不加robot
    """

    def __init__(self, config, input_dim=12, self_state_dim=5, n_actions=0, device=None):
        super().__init__()
        name = "esa2"
        self.time_step = 0.25
        mlp11_dims = [int(x) for x in config.get(name, "mlp11_dims").split(", ")]
        mlp_final_dims = [int(x) for x in config.get(name, "mlp_final").split(", ")]
        lstm_hidden_dim = int(config.get(name, "hn_state_dim"))
        mlp21_dims = [int(x) for x in config.get(name, "mlp21_dims").split(", ")]
        mlp23_dims = [int(x) for x in config.get(name, "mlp23_dims").split(", ")]
        attention_dims = [int(x) for x in config.get(name, "attention_dims").split(", ")]
        self.with_om = config.getboolean(name, "with_om")
        with_global_state = config.getboolean(name, "with_global_state")

        # print the model structure
        # summary(self.model, torch.zeros([1, 5, 13]))

        self.multiagent_training = config.getboolean(name, "multiagent_training")
        self.current_dist_weight = config.getfloat(name, "current_dist_weight")

        # SARL 网络定义
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp21_dims[-1]
        self.mlp21 = mlp(input_dim, mlp21_dims, last_relu=True)
        mlp3_input_dim = mlp21_dims[-1] + self.self_state_dim
        self.mlp23 = mlp(mlp3_input_dim, mlp23_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp21_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp21_dims[-1], attention_dims)

        # LSTM-RL 网络定义
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_dim - self_state_dim, lstm_hidden_dim, batch_first=True)
        self.mlp11 = mlp(self_state_dim + lstm_hidden_dim, mlp11_dims)

        # 预测value
        self.mlp_final2 = mlp(mlp11_dims[-1] + mlp23_dims[-1], mlp_final_dims + [n_actions])

        self.device = device

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, : self.self_state_dim]
        mlp1_output = self.mlp21(state.view((-1, size[2])))
        # mlp2_output = self.mlp22(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = (
                global_state.expand((size[0], size[1], self.global_state_dim))
                .contiguous()
                .view(-1, self.global_state_dim)
            )
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        weights = softmax(scores, dim=1).unsqueeze(2)
        # output feature is a linear combination of input features
        features = mlp1_output.view(size[0], size[1], -1)
        # for converting to onnx
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        x2 = self.mlp23(joint_state)

        soted_state = self.sort_states(state)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(soted_state[:, :, self.self_state_dim :], (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        x1 = self.mlp11(joint_state)

        values = self.mlp_final2(torch.cat([x1, x2], dim=1))

        return pfrl.action_value.DiscreteActionValue(values)

    def sort_states(self, states_batch):
        sorted_batch = []
        for batch in states_batch:
            b = batch.numpy()
            sb = sorted(b, key=self.dist, reverse=True)
            sorted_batch.append(sb)
        return torch.from_numpy(np.array(sorted_batch))

    # def dist(self, state):
    #     if np.all(state[3:5]) == 0:
    #         return np.Inf
    #
    #     # sort human order by decreasing distance to the robot
    #     current_dist = (state[3] ** 2 + state[4] ** 2) ** 0.5
    #     # human's future possible position
    #     fhx, fhy = (
    #         self.time_step * state[5] + state[3],
    #         self.time_step * state[6] + state[4],
    #     )
    #     next_possible_dist = (fhx**2 + fhy**2) ** 0.5
    #     return (
    #         self.current_dist_weight * current_dist
    #         + (1 - self.current_dist_weight) * next_possible_dist
    #     )

    def dist(self, state):
        if np.all(state[5:7]) == 0:
            return np.Inf

        # sort human order by decreasing distance to the robot
        current_dist = (state[5] ** 2 + state[6] ** 2) ** 0.5
        # human's future possible position
        fhx, fhy = (
            self.time_step * state[7] + state[5],
            self.time_step * state[8] + state[6],
        )
        next_possible_dist = (fhx**2 + fhy**2) ** 0.5
        return self.current_dist_weight * current_dist + (1 - self.current_dist_weight) * next_possible_dist
