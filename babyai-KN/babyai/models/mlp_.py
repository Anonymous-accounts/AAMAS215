import torch.nn as nn

import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, out_dim, use_orthogonal=True, use_ReLU=True):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        self.out_dim = out_dim

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc_n = nn.Sequential(
            init_(nn.Linear(hidden_size, out_dim)), active_func, nn.LayerNorm(out_dim))

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N - 1):
            x = self.fc2[i](x)
        x_n_1 = x
        x = self.fc_n(x)
        return x, x_n_1


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x


if __name__ == '__main__':
    import torch
    from babyai.arguments import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    args.hidden2_dim = 128
    args.obs_dim = args.hidden2_dim
    args.hidden_size = 64
    args.layer_N = 2
    args.out_dim = 3
    mlp_ = MLPLayer(args.obs_dim, args.hidden_size, args.layer_N, args.out_dim)
    x = torch.rand(100, args.hidden2_dim)
    y = mlp_(x)
    _, preds = torch.max(y, 1)

    labels = torch.randint(0, 3, size=(100,))

    # define loss function 
    criterion = nn.CrossEntropyLoss() 
    loss_cls = criterion(y, labels)

    print(loss_cls)



