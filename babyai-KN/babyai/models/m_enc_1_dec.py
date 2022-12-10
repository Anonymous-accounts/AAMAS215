import torch
import torch.nn as nn
from babyai.models.encoder import *

from torchkit import pytorch_utils as ptu


class one_encoder(nn.Module):
    def __init__(self, args):
        super(one_encoder, self).__init__()
        self.encoder = MLPEncoder(
                hidden_size_lst=args.hidden_size_lst, # 256
                # num_hidden_layers=1,
                task_embedding_size=args.task_embedding_size, # 8
                action_size=args.n_actions, # 7
                state_size=args.obs_dim, # 20
                reward_size=1,
                term_size=0, # encode (s,a,r,s') only
                normalize_z=args.normalize_z, # True
                recon_flag=True
        	).to(ptu.device)

    def forward(self, obs, action, reward, next_obs=None, term=None):
        if next_obs == None:
            return self.encoder(obs, action, reward)     
        if term == None:
            return self.encoder(obs, action, reward, next_obs)
        else:
            return self.encoder(obs, action, reward, next_obs, term)
        

class m_encoder_1_decoder(nn.Module):
    def __init__(self, args):
        super(m_encoder_1_decoder, self).__init__()
        self.base_task = args.num_tasks

        self.encoders = torch.nn.ModuleList([one_encoder(args) for i in range(self.base_task)]) # As many encoders as there are tasks
        self.decoder = MLPDecoder(
                input_size=args.task_embedding_size,
                hidden_size_lst=[64, 128],
                normalize_z=False,
                obs_size=args.obs_dim
            ).to(ptu.device)
    
    def forward(self, obs, action, rew):
        out = [i(obs, action, rew) for i in self.encoders]
        out = torch.stack(out, dim=1) # bs, len_tasks_num, 5
        return out