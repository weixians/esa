import torch
import torch.nn as nn
from torch.nn.functional import softmax

import pfrl
from crowd_nav.policy.cadrl import mlp


class SarlModel(nn.Module):
    def __init__(self, config, input_dim=12, self_state_dim=5, n_actions=0, device=None):
        super().__init__()
        name = "sarl2"
        mlp1_dims = [int(x) for x in config.get(name, "mlp1_dims").split(", ")]
        mlp2_dims = [int(x) for x in config.get(name, "mlp2_dims").split(", ")]
        mlp3_dims = [int(x) for x in config.get(name, "mlp3_dims").split(", ")]
        attention_dims = [int(x) for x in config.get(name, "attention_dims").split(", ")]
        self.with_om = config.getboolean(name, "with_om")
        with_global_state = config.getboolean(name, "with_global_state")

        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims, last_relu=True)
        self.mlp_values = mlp(mlp3_dims[-1], [n_actions])
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        # (batch_size, 人数, 特征维度)
        size = state.shape
        self_state = state[:, 0, : self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

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
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        out = self.mlp3(joint_state)
        values = self.mlp_values(out)

        return pfrl.action_value.DiscreteActionValue(values)
