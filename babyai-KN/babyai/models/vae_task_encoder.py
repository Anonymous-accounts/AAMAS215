import torch
import numpy as np
import torch.nn as nn
from torchkit.modules import LayerNorm

def _init_layer(m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

def build_sequential(num_inputs, hiddens, activation="relu", layer_norm=True, output_activation=False):
    modules = []

    layer_norm = layer_norm

    if activation == "relu":
        nonlin = nn.ReLU
    elif activation == "tanh":
        nonlin = nn.Tanh
    else:
        raise ValueError(f"Unknown activation option {activation}!")
    
    assert len(hiddens) > 0
    modules.append(_init_layer(nn.Linear(num_inputs, hiddens[0])))
    
    if layer_norm:
        ln = LayerNorm(hiddens[0])
        modules.append(ln)

    for i in range(len(hiddens) - 1):
        modules.append(nonlin())
        modules.append(_init_layer(nn.Linear(hiddens[i], hiddens[i + 1])))
        if layer_norm:
            ln = LayerNorm(hiddens[i+1])
            modules.append(ln)
    if output_activation:
        modules.append(nonlin())
    return nn.Sequential(*modules)


class VAETaskEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, rew_dim, task_embedding_dim, hiddens, activation, layer_norm):
        super(VAETaskEncoder, self).__init__()
        self.use_rnn = False
        input_dim = obs_dim + act_dim + rew_dim
        self.task_emb_dim = task_embedding_dim
        self.layer_norm = layer_norm

        
        if self.use_rnn:
            self.rnn_hidden_dim = hiddens[0]
            if len(hiddens) > 1:
                # at least 2 hidden layers --> 1 hidden layer before RNN, rest after
                self.input_fc = build_sequential(input_dim, [hiddens[0]], activation, output_activation=True, layer_norm=self.layer_norm)
                self.rnn = nn.GRU(input_size=hiddens[0], hidden_size=hiddens[0], num_layers=1) # TODO: RNN 是否需要呢？
                self.output_fc = build_sequential(hiddens[0], hiddens[1:] + [task_embedding_dim * 2], activation, layer_norm=self.layer_norm)
            else:
                # only 1 hidden layer --> first RNN before rest
                self.input_fc = None
                self.rnn = nn.GRU(input_size=input_dim, hidden_size=hiddens[0], num_layers=1)
                self.output_fc = build_sequential(hiddens[0], hiddens + [task_embedding_dim * 2], activation)
        else:
            self.input_fc = build_sequential(input_dim, [hiddens[0]], activation, output_activation=True)
            self.output_fc = build_sequential(hiddens[0], hiddens[1:] + [task_embedding_dim * 2], activation, output_activation=True)

        
    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def reparameterise(self, mu, log_var):
        """
        Get VAE latent sample from distribution
        :param mu: mean for encoder's latent space
        :param log_var: log variance for encoder's latent space
        :return: sample of VAE distribution
        """
        # compute standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        # get random sample with same dim as std
        eps = torch.randn_like(std)
        # sample from latent space
        sample = mu + (eps * std)
        return sample

    def forward(self, obs, act, rew, hiddens=None):
        x = torch.cat([obs, act, rew], dim=-1)

        if self.input_fc is not None:
            x = self.input_fc(x)

        # reshape to (seq_length, N, *)
        if len(x.shape) < 3:
            # sequence length missing
            seq_length = 1
            batch_size = x.shape[0]
            x_shape = x.shape[1:]
            x = x.unsqueeze(0)
            if hiddens is not None:
                hiddens_shape = hiddens.shape[1:]
                hiddens = hiddens.unsqueeze(0)
            else:
                hiddens_shape = None
        else:
            seq_length = None
            batch_size = None
            hiddens_shape = None
            x_shape = None

        if self.use_rnn:
            output, final_hiddens = self.rnn(x, hiddens)
        else:
            output = x

        # if reshaped before, unflatten again
        if seq_length is not None:
            if seq_length == 1:
                output = output.squeeze(0)
                if self.use_rnn:
                    final_hiddens = final_hiddens.squeeze(0)

        x = self.output_fc(output)

        # get mu and log_var from output
        if x.dim() > 2:
            mu = x[:, :, :self.task_emb_dim]
            log_var = x[:, :, self.task_emb_dim:]
        else:
            mu = x[:, :self.task_emb_dim]
            log_var = x[:, self.task_emb_dim:]

        # task embedding as concat of mu and log var
        task_emb = torch.cat([mu, log_var], dim=-1)

        # get sample from latent space
        z = self.reparameterise(mu, log_var)

        if self.use_rnn:
            return task_emb, mu, log_var, z, final_hiddens
        else:
            return task_emb, mu, log_var, z


class TaskDecoder(nn.Module):
    def __init__(self, task_embedding_dim, obs_dim, act_dim, rew_dim, activation):
        super(TaskDecoder, self).__init__()
        self.obs_dim = obs_dim
        self.rew_dim = rew_dim
        self.decoder = build_sequential(task_embedding_dim + obs_dim + act_dim, [obs_dim + rew_dim], activation)

    def forward(self, task_emb, obs, act):
        x = torch.cat([task_emb, obs, act], dim=-1)
        out = self.decoder(x)
        if out.dim() == 2:
            obs_pred = out[:, :self.obs_dim]
            rew_pred = out[:, self.obs_dim:]
        elif out.dim() == 3:
            obs_pred = out[:, :, :self.obs_dim]
            rew_pred = out[:, :, self.obs_dim:]

        return obs_pred, rew_pred

class VAE(nn.Module):
    def __init__(self, task_emb_dim, obs_dim, act_dim, rew_dim, n_agents, hiddens, activation, layer_norm):
        super(VAE, self).__init__()
        self.encoder = VAETaskEncoder(obs_dim, act_dim, rew_dim, task_emb_dim, hiddens, activation, layer_norm)
        self.decoder = TaskDecoder(task_emb_dim, obs_dim, act_dim, n_agents, activation)


if __name__ == '__main__':
    obs_dim = 147
    act_dim = 7
    rew_dim = 1
    n_agents = 1
    task_emb_dim = 10
    hiddens = [64]
    hidden_size = 64
    activation = 'relu'
    lr = 0.0001
    obs_loss_coef, rew_loss_coef, kl_loss_coef = 1.0, 1.0, 0.1

    vae_task_encoder = VAETaskEncoder(obs_dim, act_dim, rew_dim, task_emb_dim, hiddens, activation)
    decoder = TaskDecoder(task_emb_dim, obs_dim, act_dim, n_agents, activation)
    
    obs = torch.rand(10, obs_dim)
    act_onehot = torch.rand(10, act_dim)
    rew = torch.rand(10, rew_dim)
    hidden = torch.rand(10, hidden_size)
    joint_next_obss = torch.rand(10, obs_dim)
    joint_rews = torch.rand(10, rew_dim)

    task_emb, mu, log_var, z = vae_task_encoder(obs, act_onehot, rew)
    task_emb_train = z
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    

    joint_pred_obss, joint_pred_rews = decoder(task_emb_train, obs, act_onehot)
    obs_loss = (joint_next_obss - joint_pred_obss).pow(2).mean()
    rew_loss = (joint_rews - joint_pred_rews).pow(2).mean()

    params = list(vae_task_encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(params, lr)
    optimiser.zero_grad()
    loss = obs_loss_coef * obs_loss + rew_loss_coef * rew_loss + kl_loss_coef * kl_loss
    loss.backward()
    optimiser.step()












    

