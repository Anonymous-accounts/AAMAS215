from ast import arg
from re import A
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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

class VGAE_batch(nn.Module):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(VGAE_batch,self).__init__()
        self.args = args
        print('self.args.input_dim, self.args.hidden1_dim', self.args.input_dim, self.args.hidden1_dim)

        self.base_gcn = GCNConv(self.args.input_dim, self.args.hidden1_dim)
        self.gcn_mean = GCNConv(self.args.hidden1_dim, self.args.hidden2_dim)
        self.gcn_logstddev = GCNConv(self.args.hidden1_dim, self.args.hidden2_dim)
        
        # self.use_bn = self.args.use_bn
        # self.
        # if self.use_bn:
        #     self.batch_norm = nn.BatchNorm1d(self.args.input_dim)
        # self.mlp_class = MLPLayer(args.obs_dim, args.hidden_size, args.layer_N, args.out_dim)

    def transfer_edge_index(self, adj_):
        ed = sp.coo_matrix(adj_.cpu())
        indices = np.vstack((ed.row, ed.col))
        index = torch.LongTensor(indices).to(self.device)
        return index

    def make_batch(self, feature_batch, edge_index_lst): # 3.4.64
        data_list = []
        follow_batch= []
        for i in range(feature_batch.shape[0]):
            data = Data(edge_index=edge_index_lst[i], x=feature_batch[i], y=torch.Tensor([i]))
            data_list.append(data)
            follow_batch.append('x_'+str(i))
        loader = DataLoader(data_list, batch_size=feature_batch.shape[0], follow_batch=follow_batch)
        for data in loader:
            x = data.x
            edge_index = data.edge_index
            return x, edge_index

        

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

        # import pdb
        # pdb.set_trace()
        hidden = F.relu(self.base_gcn(X, adj))  # torch.Size([2, 64])
        self.mean = F.relu(self.gcn_mean(hidden, adj))
        self.logstd = F.relu(self.gcn_logstddev(hidden, adj))
        gaussian_noise = torch.randn(X.size(0), self.args.hidden2_dim).to(self.device)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z # torch.Size([2, 32])

    def forward(self, X, adj):
        # import pdb
        # pdb.set_trace()
        Z = self.encode(X, adj)
        A_pred = dot_product_decode(Z)
        return Z, A_pred
    
    def forward_batch(self, X, adj):
        x, edge_index = self.make_batch(X, adj)
        hidden = F.relu(self.base_gcn(x, edge_index))  # torch.Size([2, 64])
        hidden = hidden.reshape(len(adj), -1, self.args.hidden1_dim)
        hidden, edge_index = self.make_batch(hidden, adj)
        self.mean = F.relu(self.gcn_mean(hidden, edge_index))
        self.logstd = F.relu(self.gcn_logstddev(hidden, edge_index))
        self.mean = self.mean.reshape(len(adj), -1, self.args.hidden2_dim)
        self.logstd = self.logstd.reshape(len(adj), -1, self.args.hidden2_dim)
        gaussian_noise = torch.randn(X.size(0), X.size(1), self.args.hidden2_dim).to(self.device)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        A_pred = [dot_product_decode(sampled_z[i]) for i in range(sampled_z.shape[0])]
        A_pred = torch.stack(A_pred)

        return sampled_z, A_pred
        

    def backward(self, A_pred, adj_label, norm, weight_tensor):
        log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        kl_divergence = 0.5/ A_pred.size(0) * (1 + 2 * self.logstd - self.mean**2 - torch.exp(self.logstd)**2).sum(1).mean()
        # loss = log_lik - kl_divergence
        return log_lik, -kl_divergence



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
        # # x: [batchsize, num_of_point, dim]
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

    vgae_batch = VGAE_batch(args)
    vgae_optimizer_batch = torch.optim.Adam(vgae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if torch.cuda.is_available():
        vgae.cuda()
        vgae_batch.cuda()


    features_ = torch.rand((2, 100)).float().to(vgae.device)
    adj_ = np.array([[0, 1], [0, 0]])

    # import pdb
    # pdb.set_trace()
    adj_norm, adj_label, norm, weight_tensor = vgae.data_transfer(adj_)

    import datetime
    starttime = datetime.datetime.now()
    z_lst = []
    A_pred_lst = []
    for i in range(64):
        z, A_pred = vgae(features_, adj_norm) ### 为什么变成了2维
        z_lst.append(z)
        A_pred_lst.append(A_pred)
    endtime = datetime.datetime.now()
    print('original forward 耗时：', (endtime - starttime))

    # for name, parameters in vgae.named_parameters():
    #     print(name, ':', parameters)
    #     break
    import datetime
    starttime = datetime.datetime.now()
    losses = 0
    for i in range(64):
        log_lik, kl_divergence = vgae.backward(A_pred_lst[i], adj_label, norm, weight_tensor)
        loss = log_lik + 0.1 * kl_divergence
        losses += loss
        
    vgae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
    losses.backward(retain_graph=True)  # 将误差反向传播
    vgae_optimizer.step()  # 更新参数
    print(loss)
    print(log_lik)
    print(kl_divergence)
    endtime = datetime.datetime.now()
    print('original backward 耗时：', (endtime - starttime))
    

    # import pdb
    # pdb.set_trace()
    ################################################################################################## 

    import datetime
    starttime = datetime.datetime.now()

    features_ = torch.rand((64, 2, 100)).float().to(vgae.device)
    adj_lists = [vgae_batch.transfer_edge_index(adj_) for i in range(64)]
    z, A_pred = vgae_batch.forward_batch(features_, adj_lists) ### 为什么变成了2维

    endtime = datetime.datetime.now()
    print('_batch forward 耗时：', (endtime - starttime))

    # for name, parameters in vgae.named_parameters():
    #     print(name, ':', parameters)
    #     break
    import datetime
    starttime = datetime.datetime.now()
    vgae_optimizer_batch.zero_grad()  # 在反向传播之前，先将梯度归0
    log_lik, kl_divergence = 0, 0
    for i in range(64):
        # adj_norm, adj_label, norm, weight_tensor = vgae.data_transfer(adj_)
        a, b = vgae_batch.backward(A_pred[i], adj_label, norm, weight_tensor)
        log_lik += a
        kl_divergence += b
    loss = log_lik + 0.1 * kl_divergence
    print(loss)
    print(log_lik)
    print(kl_divergence)
    loss.backward()  # 将误差反向传播
    vgae_optimizer_batch.step()  # 更新参数
    endtime = datetime.datetime.now()
    print('_batch backward 耗时：', (endtime - starttime))




