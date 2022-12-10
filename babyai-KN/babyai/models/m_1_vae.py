from ast import arg
from cProfile import label
from tarfile import ENCODING
import torch
import torch.nn as nn

class one_encoder(nn.Module):
    def __init__(self, args):
        super(one_encoder, self).__init__()
        self.state_input_dim = args.latent_dim * args.n_agents
        self.action_input_dim = args.n_actions
        self.reward_input_dim = 1
        self.latent_dim = args.latent_dim * args.n_agents
        self.hidden_dim = args.hidden_dim
        self.n_agents = args.n_agents
        self.action_embedding_dim = 8
        self.reward_embedding_dim = 4
        self.state_input_dim_ = self.state_input_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_input_dim, self.action_embedding_dim)
        )

        self.reward_encoder = nn.Sequential(
            nn.Linear(self.reward_input_dim, self.reward_embedding_dim)
        )

        self.next_state_encoder = nn.Sequential(
            nn.Linear(self.state_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),
        )

        self.cat_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.action_embedding_dim * args.n_agents + \
                      self.reward_embedding_dim * args.n_agents, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_input_dim_)
        )

        self.logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_input_dim_)
        )
        
    def forward(self, s_t_1, a, r, s_t_2):
        state_encoder = self.state_encoder(s_t_1) # 32.128
        action_encoder = self.action_encoder(a) # 32.8
        reward_encoder = self.reward_encoder(r) # 32.8
        next_state_encoder = self.next_state_encoder(s_t_2) # 32.128
        encoder = torch.cat([state_encoder, action_encoder, reward_encoder, next_state_encoder], dim=-1)
        encoder = self.cat_encoder(encoder)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar



class _autoencoder(nn.Module):
    def __init__(self, args):
        super(_autoencoder, self).__init__()
        self.state_input_dim = args.latent_dim * args.n_agents
        self.action_input_dim = args.n_actions
        self.reward_input_dim = 1
        self.latent_dim = args.latent_dim * args.n_agents
        self.hidden_dim = args.hidden_dim
        self.action_embedding_dim = 8
        self.reward_embedding_dim = 4
        self.n_agents = args.n_agents
        self.state_input_dim_ = self.state_input_dim
        self.base_task = 5

        self.encoders = torch.nn.ModuleList([one_encoder(args) for i in range(self.base_task)]) # As many encoders as there are tasks

        # self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_component, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.pi, 1. / self.n_component, 1. / self.n_component + 0.01)

        self.state_decoder = nn.Sequential(
            nn.Linear(self.state_input_dim_, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_input_dim),
            nn.Softmax()
        )

        self.next_state_decoder = nn.Sequential(
            nn.Linear(self.state_input_dim_, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_input_dim),
            nn.Softmax()
        )

        self.action_decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(self.state_input_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_input_dim),
            nn.Softmax()
        ) for _ in range(args.n_agents)])

        self.reward_decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(self.state_input_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax()
        ) for _ in range(args.n_agents)])

    
    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z):
        state_decoder = self.state_decoder(z)
        next_state_decoder = self.next_state_decoder(z)
        action_decoders, reward_decoders = [], []
        for n in range(self.n_agents):
            action_decoder = self.action_decoder[n](z)
            reward_decoder = self.reward_decoder[n](z)
            action_decoders.append(action_decoder)
            reward_decoders.append(reward_decoder)
        action_decoders = torch.stack(action_decoders, dim=-2)[:, 0]
        reward_decoders = torch.stack(reward_decoders, dim=-2)[:, 0]
        # action_decoders = torch.log_softmax(action_decoders, dim=-1)
        # reward_decoders = torch.log_softmax(reward_decoders, dim=-1)
        return state_decoder, action_decoders, reward_decoders, next_state_decoder

    def forward(self, s_t_1, a, r, s_t_2, labels):
        z = torch.zeros((labels.shape[0], self.encoders[0].state_input_dim_))
        mu = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        for i in range(self.base_task):
            index_i = torch.where(labels == i)[0]
            if len(index_i) != 0:
                s_t_1_i, a_i, r_i, s_t_2_i = s_t_1[index_i], a[index_i], r[index_i], s_t_2[index_i]
                mu_i, logvar_i = self.encoders[i](s_t_1_i, a_i, r_i, s_t_2_i)
                z_i = self.sample_z(mu_i, logvar_i)
                z[index_i], mu[index_i], logvar[index_i] = z_i, mu_i, logvar_i

        state_decoder, action_decoders, reward_decoders, next_state_decoder = self.decode(z)
        return state_decoder, action_decoders, reward_decoders, next_state_decoder, mu, logvar, z

    def backward(self, s_t_1, a, r, s_t_2, s_t_1_, a_, r_, s_t_2_, mean, std):
        # import pdb
        # pdb.set_trace()
        # BCE = torch.tensor([torch.nn.functional.binary_cross_entropy(s_t_1_[i], s_t_1[i], reduction='sum') for i in range(s_t_1.shape[0])]).mean()
        # BCE += torch.tensor([torch.nn.functional.binary_cross_entropy(a_[i], a[i], reduction='sum') for i in range(s_t_1.shape[0])]).mean()
        # BCE += torch.tensor([torch.nn.functional.binary_cross_entropy(r_[i], r[i], reduction='sum') for i in range(s_t_1.shape[0])]).mean()
        # BCE += torch.tensor([torch.nn.functional.binary_cross_entropy(s_t_2_[i], s_t_2[i], reduction='sum') for i in range(s_t_1.shape[0])]).mean()
        BCE = torch.nn.functional.binary_cross_entropy(s_t_1_, s_t_1, reduction='mean')
        BCE += torch.nn.functional.binary_cross_entropy(a_, a, reduction='mean')
        BCE += torch.nn.functional.binary_cross_entropy(r_, r, reduction='mean')
        BCE += torch.nn.functional.binary_cross_entropy(s_t_2_, s_t_2, reduction='mean')
        # BCE = (s_t_1 - s_t_1_).pow(2).sum()
        # BCE += (a - a_).pow(2).sum()
        # BCE += (r - r_).pow(2).sum()
        # BCE += (s_t_2 - s_t_2_).pow(2).sum()
        # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
        var = torch.pow(torch.exp(std), 2)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        # -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss_all = BCE + KLD  # TODO:0.1 # decouple with parameter > 1
        return loss_all, BCE, KLD


if __name__ == '__main__':
    # Parse arguments
    from babyai.arguments import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    args.latent_dim = 100
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 128
    vae = _autoencoder(args)

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    s_t_1 = torch.rand(4, 100)
    a = torch.rand(4, 7)
    r = torch.rand(4, 1)
    s_t_2 = torch.rand(4, 100)
    labels = torch.tensor([0,1,2,4])

    s_t_1_, a_, r_, s_t_2_, mu, var, z = vae(s_t_1, a, r, s_t_2, labels)
    # 32.100, 32.7, 32.1, 32.100, 32.100, 32.100
    # print(s_1_, a_, r_, s_2_)
    loss_all, BCE, KLD = vae.backward(s_t_1, a, r, s_t_2, s_t_1_, a_, r_, s_t_2_, mu, var)

    for name, parameters in vae.named_parameters():
        print(name, ':', parameters)
        break

    vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
    loss_all.backward()  # 将误差反向传播
    vae_optimizer.step()  # 更新参数

    for name, parameters in vae.named_parameters():
        print(name, ':', parameters)
        break







