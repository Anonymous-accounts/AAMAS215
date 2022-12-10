from ast import arg
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import scipy.sparse as sp
# from scipy import sparse
from babyai.models.vgae_processing import *
from babyai.models.mlp_ import MLPLayer
# import args


class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        print('self.args.input_dim, self.args.hidden1_dim', self.args.input_dim, self.args.hidden1_dim)

        self.base_gcn = GraphConvSparse(self.args.input_dim, self.args.hidden1_dim)
        self.gcn_mean = GraphConvSparse(self.args.hidden1_dim, self.args.hidden2_dim)
        self.gcn_logstddev = GraphConvSparse(self.args.hidden1_dim, self.args.hidden2_dim)
        self.use_bn = self.args.use_bn
        # self.
        # if self.use_bn:
        #     self.batch_norm = nn.BatchNorm1d(self.args.input_dim)
        # self.mlp_class = MLPLayer(args.obs_dim, args.hidden_size, args.layer_N, args.out_dim)

        

    def data_transfer(self, adj):
        """
        adj : n * n
        node_features : n * emd_dim
        """
        
        # features = sp.csr_matrix(node_features)
        adj = sp.csr_matrix(adj)
        # Store original adjacency matrix (without diagonal entries) for later
        # adj_orig = adj
        # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        # adj_orig.eliminate_zeros()

        # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        # adj = adj_train
        adj_train = adj

        # Some preprocessing
        adj_norm = preprocess_graph(adj)

        num_nodes = adj.shape[0]  # 2708

        # features = sparse_to_tuple(features.tocoo()) # (zize:(49216, 2), ) size:(49216,)  value:(2708, 1433))
        # num_features = features[2][1]
        # features_nonzero = features[1].shape[0]

        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()                      # 815.9857397504456

        if float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) == 0:
            norm = float(adj.shape[0] * adj.shape[0])
        else:
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)    # 0.5006127558064347


        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)



        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                    torch.FloatTensor(adj_norm[1]),
                    torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                    torch.FloatTensor(adj_label[1]),
                    torch.Size(adj_label[2]))
        # features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
        #             torch.FloatTensor(features[1]),
        #             torch.Size(features[2]))

        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight


        return adj_norm.to(self.device), adj_label.to(self.device), norm, weight_tensor.to(self.device)
    
    def encode(self, X, adj):
        # # x: [num_of_point, dim]
        # if self.use_bn:
        #     X = X.unsqueeze(2)
        #     X = X.transpose(0, 2)             # # x: [1, dim， num_of_point]
        #     X = self.batch_norm(X)
        #     X = X.transpose(0, 2).squeeze()   # # x: [num_of_point, dim]

        hidden = self.base_gcn(X, adj)  # torch.Size([2, 64])
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(X.size(0), self.args.hidden2_dim).to(self.device)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z # torch.Size([2, 32])

    def forward(self, X, adj):
        # import pdb
        # pdb.set_trace()
        Z = self.encode(X, adj)
        A_pred = dot_product_decode(Z)
        return Z, A_pred

    def backward(self, A_pred, adj_label, norm, weight_tensor):
        log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) * (1 + 2 * self.logstd - self.mean**2 - torch.exp(self.logstd)**2).sum(1).mean()
        # loss = log_lik - kl_divergence
        return log_lik, -kl_divergence


    
class GAE(nn.Module):
    def __init__(self, args):
        super(GAE,self).__init__()
        self.base_gcn = GraphConvSparse(self.args.input_dim, self.args.hidden1_dim)
        # self.gcn_mean = GraphConvSparse(self.args.hidden1_dim, self.args.hidden2_dim, activation=lambda x:x)
        self.gcn_mean = GraphConvSparse(self.args.hidden1_dim, self.args.hidden2_dim)

    def encode(self, X, adj):
        hidden = self.base_gcn(X, adj)
        z = self.mean = self.gcn_mean(hidden, adj)
        return z

    def forward(self, X, adj):
        Z = self.encode(X, adj)
        A_pred = dot_product_decode(Z)
        return Z, A_pred





class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        # self.adj = adj
        # self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(adj, x)
        outputs = F.relu(x)
        return outputs

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)








if __name__ == '__main__':
    # Parse arguments
    from babyai.arguments import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    args.input_dim = 100
    args.n_agents = 1
    args.n_actions = 7
    args.hidden1_dim = 64
    args.hidden2_dim = 32
    args.use_bn = True

    vgae = VGAE(args)
    vgae_optimizer = torch.optim.Adam(vgae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    features_ = torch.rand((2, 100)).float().to(vgae.device)
    adj_ = np.array([[0, 1], [0, 0]])

    import pdb
    pdb.set_trace()
    adj_norm, adj_label, norm, weight_tensor = vgae.data_transfer(adj_)

    z, A_pred = vgae(features_, adj_norm) ### 为什么变成了2维


    # for name, parameters in vgae.named_parameters():
    #     print(name, ':', parameters)
    #     break

    vgae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
    log_lik, kl_divergence = vgae.backward(A_pred, adj_label, norm, weight_tensor)
    loss = log_lik + 0.1 * kl_divergence
    print(loss)
    print(log_lik)
    print(kl_divergence)
    loss.backward()  # 将误差反向传播
    vgae_optimizer.step()  # 更新参数



    # features_ = np.random.random((3, 100))
    # adj_ = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]])

    # adj_norm, adj_label, features, norm, weight_tensor = data_transfer(adj_, features_)

    # z, A_pred = vgae(features, adj_norm) ### 为什么变成了2维


    # vgae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
    # log_lik, kl_divergence = vgae.backward(A_pred, adj_label, norm, weight_tensor)
    # loss = log_lik + 0.1 * kl_divergence
    # print(loss)
    # print(log_lik)
    # print(kl_divergence)
    # loss.backward()  # 将误差反向传播
    # vgae_optimizer.step()  # 更新参数



    # torch.save(vgae, './vgae.pt')


    # for name, parameters in vgae.named_parameters():
    #     print(name, ':', parameters)
    #     break



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class state_autoencoder(nn.Module):
#     def __init__(self, args):
#         super(state_autoencoder, self).__init__()
#         self.args = args
#         #self.input_dim = int(np.prod(args.observation_shape)) + args.n_actions
#         self.input_dim = args.rnn_hidden_dim
#         self.latent_dim = args.latent_dim
#         self.hidden_dim = args.hidden_dim
#         self.n_agents = args.n_agents

#         self.encoder_weight_1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
#         #self.gru_layer = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
#         self.weight_logvar = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
#         self.weight_mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)

#         self.decoder_weight_1 = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
#         self.decoder_weight_2 = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

#     def encode(self, node_features, adj_list):
#         hidden = torch.zeros(1, node_features.size(0) * node_features.size(1), self.hidden_dim).to(node_features.device)

#         a_tilde = adj_list + torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand_as(adj_list).to(adj_list.device)
#         d_tilde_diag = a_tilde.sum(-1).view(-1, self.n_agents, 1) ** (-0.5)
#         d_tilde = d_tilde_diag.bmm(torch.ones_like(d_tilde_diag).permute([0, 2, 1])) * torch.eye(self.n_agents).unsqueeze(0).to(a_tilde.device)
#         encoder_factor = d_tilde.bmm(a_tilde.view(-1, self.n_agents, self.n_agents)).bmm(d_tilde)

#         node_features = node_features.reshape([-1, self.n_agents, self.input_dim])

#         encoder_1 = self.encoder_weight_1(encoder_factor.bmm(node_features))
#         encoder_1 = F.relu(encoder_1)
#         #encoder_1, _ = self.gru_layer(encoder_1, hidden)

#         encoder_2 = encoder_factor.bmm(encoder_1)

#         mu = self.weight_mu(encoder_2).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.latent_dim)
#         logvar = self.weight_logvar(encoder_2).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.latent_dim)
#         return mu, logvar

#     def sample_z(self, mu, logvar):
#         eps = torch.randn_like(logvar)
#         return mu + eps * torch.exp(0.5 * logvar)

#     def decode(self, z, adj_list):
#         a_hat = -adj_list + 2 * torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand_as(adj_list).to(adj_list.device)
#         d_hat_diag = (adj_list.sum(-1).view(-1, self.n_agents, 1) + 2) ** (-0.5)
#         d_hat = d_hat_diag.bmm(torch.ones_like(d_hat_diag).permute([0, 2, 1])) * torch.eye(self.n_agents).unsqueeze(0).to(a_hat.device)
#         decoder_factor = d_hat.bmm(a_hat.view(-1, self.n_agents, self.n_agents)).bmm(d_hat)

#         z = z.view(-1, self.n_agents, self.latent_dim)

#         decoder_1 = self.decoder_weight_1(decoder_factor.bmm(z))
#         decoder_1 = F.relu(decoder_1)

#         recon_node_features = self.decoder_weight_2(decoder_factor.bmm(decoder_1)).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.input_dim)
#         return recon_node_features

#     def forward(self, node_features, adj_list):
#         adj_list = adj_list * (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0).to(adj_list.device)
#         mu, logvar = self.encode(node_features, adj_list)
#         z = self.sample_z(mu, logvar)
#         recon_node_features = self.decode(z, adj_list)
#         return recon_node_features, mu.view(adj_list.size(0), adj_list.size(1), -1), logvar.view(adj_list.size(0), adj_list.size(1), -1), z.view(adj_list.size(0), adj_list.size(1), -1)


# class flatten_state_autoencoder(nn.Module):
#     def __init__(self, args):
#         super(flatten_state_autoencoder, self).__init__()
#         self.args = args
#         self.input_dim = args.rnn_hidden_dim
#         self.latent_dim = args.latent_dim
#         self.hidden_dim = args.hidden_dim
#         self.n_agents = args.n_agents

#         self.encoder_weight_1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.weight_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
#         self.weight_mu = nn.Linear(self.hidden_dim, self.latent_dim)

#         self.decoder_weight_1 = nn.Linear(self.latent_dim, self.hidden_dim)
#         self.decoder_weight_2 = nn.Linear(self.hidden_dim, self.input_dim)

#     def encode(self, node_features, adj_list):
#         bs = adj_list.size(0)
#         sl = adj_list.size(1)
#         encoder_1 = F.relu(self.encoder_weight_1(node_features))
#         mu = self.weight_mu(encoder_1).view(bs, sl, self.n_agents, self.latent_dim)
#         logvar = self.weight_logvar(encoder_1).view(bs, sl, self.n_agents, self.latent_dim)
#         return mu, logvar

#     def sample_z(self, mu, logvar):
#         eps = torch.randn_like(logvar)
#         return mu + eps * torch.exp(0.5 * logvar)

#     def decode(self, z, adj_list):
#         decoder_1 = F.relu(self.decoder_weight_1(z))
#         recon_node_features = self.decoder_weight_2(decoder_1).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.input_dim)
#         return recon_node_features

#     def forward(self, node_features, adj_list):
#         adj_list = adj_list * (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0).to(adj_list.device)
#         mu, logvar = self.encode(node_features, adj_list)
#         z = self.sample_z(mu, logvar)
#         recon_node_features = self.decode(z, adj_list)
#         return recon_node_features, mu.view(adj_list.size(0), adj_list.size(1), -1), logvar.view(adj_list.size(0), adj_list.size(1), -1), z.view(adj_list.size(0), adj_list.size(1), -1)


