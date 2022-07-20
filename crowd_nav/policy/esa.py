import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from torchsummaryX import summary


class ValueNetwork(nn.Module):
    """
    LSTM部分只编码人，不加robot
    """

    def __init__(
        self,
        input_dim,
        self_state_dim,
        mlp21_dims,
        mlp23_dims,
        attention_dims,
        with_global_state,
        cell_size,
        cell_num,
        mlp11_dims,
        lstm_hidden_dim,
        mlp_final_dims,
        device,
    ):
        super().__init__()
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
        self.cell_size = cell_size
        self.cell_num = cell_num
        self.attention_weights = None

        # LSTM-RL 网络定义
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_dim - self_state_dim, lstm_hidden_dim, batch_first=True)
        self.mlp11 = mlp(self_state_dim + lstm_hidden_dim, mlp11_dims)

        # 预测value
        self.mlp_final = mlp(mlp11_dims[-1] + mlp23_dims[-1], mlp_final_dims)

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
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp1_output.view(size[0], size[1], -1)
        # for converting to onnx
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        x2 = self.mlp23(joint_state)

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(state[:, :, self.self_state_dim :], (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        x1 = self.mlp11(joint_state)

        value = self.mlp_final(torch.cat([x1, x2], dim=1))

        return value


class ESA(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = "ESA"

    def configure(self, config):
        self.set_common_parameters(config)
        mlp11_dims = [int(x) for x in config.get("esa", "mlp11_dims").split(", ")]
        mlp_final_dims = [int(x) for x in config.get("esa", "mlp_final").split(", ")]
        hn_state_dim = int(config.get("esa", "hn_state_dim"))
        mlp21_dims = [int(x) for x in config.get("esa", "mlp21_dims").split(", ")]
        mlp23_dims = [int(x) for x in config.get("esa", "mlp23_dims").split(", ")]
        attention_dims = [int(x) for x in config.get("esa", "attention_dims").split(", ")]
        self.with_om = config.getboolean("esa", "with_om")
        with_global_state = config.getboolean("esa", "with_global_state")
        self.model = ValueNetwork(
            self.input_dim(),
            self.self_state_dim,
            mlp21_dims,
            mlp23_dims,
            attention_dims,
            with_global_state,
            self.cell_size,
            self.cell_num,
            mlp11_dims,
            hn_state_dim,
            mlp_final_dims,
            self.device,
        )

        # print the model structure
        # summary(self.model, torch.zeros([1, 5, 13]))

        self.multiagent_training = config.getboolean("esa", "multiagent_training")
        self.current_dist_weight = config.getfloat("esa", "current_dist_weight")

        logging.info("Policy: {} {} global state".format(self.name, "w/" if with_global_state else "w/o"))
        super().configure(config)

    def get_attention_weights(self):
        return self.model.attention_weights

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            current_dist = np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))
            rpx, rpy = state.self_state.position
            fhx, fhy = (
                self.env.time_step * human.vx + human.px,
                self.env.time_step * human.vy + human.py,
            )
            next_possible_dist = np.linalg.norm(np.array([rpx, rpy]) - np.array([fhx, fhy]))
            return self.current_dist_weight * current_dist + (1 - self.current_dist_weight) * next_possible_dist

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().predict(state)
