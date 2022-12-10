import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp
from babyai.models.gat import *

class MLPEncoder(nn.Module):
    def __init__(self,
                 # network size
                #  hidden_size=64,
                 hidden_size_lst=[256, 128, 64, 8],
                #  num_hidden_layers=2,
                 task_embedding_size=32,
                 # actions, states, rewards
                 action_size=2,
                 state_size=2,
                 reward_size=1,
                 term_size=1,
                 normalize_z=False,
                 normalize=False
                 ):
        super(MLPEncoder, self).__init__()
        self.task_embedding_size=task_embedding_size * 3
        self.gat_task_embedding_size = task_embedding_size
        self.action_size=action_size
        self.state_size=state_size
        self.reward_size=reward_size
        self.term_size=term_size
        self.normalize_z = normalize_z

        self.encoder = FlattenMlp(input_size=state_size*2+action_size+reward_size+term_size,
                                    output_size=hidden_size_lst[-1],
                                    hidden_sizes=hidden_size_lst[:-1],
                                    # layer_norm=self.normalize
                                    )
        self.encoder_2 = FlattenMlp(input_size=hidden_size_lst[-1],
                                    output_size=self.task_embedding_size,
                                    hidden_sizes=[self.task_embedding_size for i in range(1)],
                                    # layer_norm=self.normalize
                                    )

        # self.encoder_1 = FlattenMlp(input_size=state_size*2+action_size+reward_size+term_size,
        #                             output_size=256,
        #                             hidden_sizes=[256 for i in range(1)])
        # self.encoder = FlattenMlp(input_size=256,
        #                             output_size=32,
        #                             hidden_sizes=[hidden_size for i in range(num_hidden_layers)])
        # self.encoder_2 = FlattenMlp(input_size=32,
        #                             output_size=self.task_embedding_size,
        #                             hidden_sizes=[self.task_embedding_size for i in range(1)])

        #### The difference from the original is that the output of mlp here is not normalized, the output of gat is normalized, and the z-dim of the middle link between the two = ori_dim * 3
        self.gat_model = GAT(nfeat=self.task_embedding_size, 
                nhid=self.task_embedding_size,
                nout=self.gat_task_embedding_size,
                dropout=0.5, 
                nheads=1, 
                alpha=0.2,
                normalize=normalize)

        self.normalize=normalize

        self.use_termination = True if term_size else False # if term_size=0, encode (s,a,r,s') only

    
    def contruct_edges(self, batchsize, neg_size_, pos_neg_gap):
        # 64, 64, 64*5
        import itertools
        number = batchsize + batchsize + batchsize * neg_size_
        edge_index_full_connect = torch.ones(number, number)  #  # Generate a fully connected adjacency matrix
        

        postive_list = torch.arange(0, batchsize + batchsize)
        neg_list = torch.arange(batchsize + batchsize, number)
        # import pdb
        # pdb.set_trace()
        for i in postive_list: # Negative example information exchange is 0.2 ratio
            for j in neg_list:
                edge_index_full_connect[i][j] = pos_neg_gap
                edge_index_full_connect[j][i] = pos_neg_gap

        return edge_index_full_connect


    # input state transition sample, output task embedding

    def forward(self, obs, action, reward, next_obs, term=None):
        assert obs.shape[1] == self.state_size and action.shape[1] == self.action_size \
            and reward.shape[1] == self.reward_size and next_obs.shape[1] == self.state_size \
            and ((not self.use_termination) or (term.shape[1] == self.term_size))
        out = self.encoder(obs, action, reward, next_obs, term) if self.use_termination \
            else self.encoder(obs, action, reward, next_obs)
        out = self.encoder_2(out)
        
        if not self.normalize_z:
            return out
        else:
            return F.normalize(out)

    # def forward(self, obs, action, reward, next_obs, term=None):
    #     assert obs.shape[1] == self.state_size and action.shape[1] == self.action_size \
    #         and reward.shape[1] == self.reward_size and next_obs.shape[1] == self.state_size \
    #         and ((not self.use_termination) or (term.shape[1] == self.term_size))
    #     out = self.encoder_1(obs, action, reward, next_obs, term) if self.use_termination \
    #         else self.encoder_1(obs, action, reward, next_obs)
    #     out = self.encoder(out)
    #     out = self.encoder_2(out)
    #     return out 
        #### here not normalize
        # if not self.normalize:
        #     return out
        # else:
        #     return F.normalize(out)

    # extract task representation from context sequence
    # input size: (timesteps, task, dim)
    # output size: (task, z_dim)
    def context_encoding(self, obs, actions, rewards, next_obs, terms):
        n_timesteps, batch_size, _ = obs.shape
        #print(obs.shape, actions.shape, rewards.shape, next_obs.shape, terms.shape)
        z = self.forward(
                obs.reshape(n_timesteps*batch_size, -1),
                actions.reshape(n_timesteps*batch_size, -1),
                rewards.reshape(n_timesteps*batch_size, -1),
                next_obs.reshape(n_timesteps*batch_size, -1),
                terms.reshape(n_timesteps*batch_size, -1)
            )
        z = z.reshape(n_timesteps, batch_size, -1)
        z = z.mean(0) # average over timesteps
        #print(z.shape)
        return z

# encoder that convert sample encodings to a context encoding
# context {(s,a,r,s')_i} -> sample encodings {z_i} -> context encoding z
# output mlp layars are only for debug (train with task gt)
class SelfAttnEncoder(nn.Module):
    def __init__(self, input_dim=5, num_output_mlp=0, task_gt_dim=5):
        super(SelfAttnEncoder, self).__init__()
        self.input_dim = input_dim
        self.score_func = nn.Linear(input_dim, 1)
        self.num_output_mlp = num_output_mlp
        if num_output_mlp > 0:
            self.output_mlp = Mlp(input_size=input_dim,
                                output_size=task_gt_dim,
                                hidden_sizes=[64 for i in range(num_output_mlp-1)])

    # input (b, N, dim), output (b, dim)
    def forward(self, inp):
        b, N, dim = inp.shape
        scores = self.score_func(inp.reshape(-1, dim)).reshape(b, N)
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context

    def forward_full(self, inp):
        context = self.forward(inp)
        if self.num_output_mlp > 0:
            task_pred = self.output_mlp(context)
            return context, task_pred
        else:
            return context


# encoder that takes average pooling
# context {(s,a,r,s')_i} -> sample encodings {z_i} -> context encoding z
class MeanEncoder(nn.Module):
    def __init__(self):
        super(MeanEncoder, self).__init__()

    def forward(self, inp):
        b, N, dim = inp.shape
        z = inp.mean(1)
        #print(z.shape)
        return z





if __name__ == '__main__':
    from babyai.arguments import ArgumentParser
    # Parse arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    print('initialize...')
    args.aggregator_hidden_size = 128
    args.task_embedding_size = 5
    args.obs_dim = 147
    args.n_actions = 7
    args.normalize_z = True
    args.encoder_lr = 0.0003
    args.contrastive_batch_size = 64
    args.n_negative_per_positive = 16
    args.reward_std = 0.5
    args.infonce_temp = 0.1
    args.sizes = [args.obs_dim, args.n_actions, 1, args.obs_dim, 1]

    encoder = MLPEncoder(
                hidden_size=args.aggregator_hidden_size, # 256
                num_hidden_layers=2,
                task_embedding_size=args.task_embedding_size, # 5
                action_size=args.n_actions, # 7
                state_size=args.obs_dim, # 20
                reward_size=1,
                term_size=0, # encode (s,a,r,s') only
                normalize=args.normalize_z # True
        	)
    
    obs_q = torch.rand(2, 147)
    actions_q = torch.rand(2, 7)
    rewards_q = torch.rand(2, 1)
    next_obs_q = torch.rand(2, 147)
    q_z = encoder.forward(obs_q, actions_q, rewards_q, next_obs_q) # 64, 5
    k_z = encoder.forward(obs_q, actions_q, rewards_q, next_obs_q) # 64, 5
    
    # obs_q = torch.rand(2, 4, 147)
    # actions_q = torch.rand(2, 4, 7)
    # rewards_q = torch.rand(2, 4, 1)
    # next_obs_q = torch.rand(2, 4, 147)
    # neg_z = encoder.forward(obs_q, actions_q, rewards_q, next_obs_q) # 64, 4, 5
    q_z = encoder.forward(obs_q, actions_q, rewards_q, next_obs_q) # 64, 5
    neg_z = q_z.unsqueeze(1).repeat_interleave(4, 1)

    att = GAT(nfeat=5, 
                nhid=64,
                nout=5,
                dropout=0.5, 
                nheads=2, 
                alpha=0.2)

    batchsize = 2
    neg_size_ = 4
    pos_neg_gap = 0.2
    adj = contruct_edges(batchsize, neg_size_, pos_neg_gap)
    print(adj.shape)
    

    xx = torch.cat((q_z, k_z, neg_z.reshape(-1, args.task_embedding_size)), dim=0)
    print(xx.shape)

    yy = att(xx, adj)
    print(yy.shape)
    
    
