#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import numpy as np
import torch

import babyai
import babyai.utils as utils
from babyai.arguments import ArgumentParser
from babyai.models.vae import _autoencoder

from utils.recursive_read import return_npy_events_list
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
# Figure out what's the problem

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()
args.save_logdir = './logs_models_v1/'
args.n_epoch = 30
args.batch_size_ = 32

cur_time = datetime.now() + timedelta(hours=0)
args.log_time = cur_time.strftime("[%m-%d]%H.%M.%S")
writer = SummaryWriter(logdir=args.save_logdir + '/' + args.log_time + "/logs")

utils.seed(args.seed)

obss_preprocessor = utils.ObssPreprocessor(args.save_logdir, None, None)


def trans_image(image):
    return torch.transpose(torch.transpose(image, 1, 3), 2, 3)



if __name__ == '__main__':
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 128
    vae = _autoencoder(args)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print('begin to train the task representation')
    
    loader_path = '../datasets/train_loader_bs64'
    train_loader = torch.load(loader_path)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(train_loader):  # 100*32
            print("epoch, minibatch of traning vae ------", epoch, i)
            s_t_1, a, r, s_t_2 = images[:, :147], images[:, 147:147+args.n_actions], images[:, [147++args.n_actions]], images[:, -147:] # s,a,r,s'
            s_t_1_, a_, r_, s_t_2_, mu, var, z = vae(s_t_1, a, r, s_t_2)
            # 32.100, 32.7, 32.1, 32.100, 32.100, 32.100
            loss, BCE, KLD = vae.backward(s_t_1, a, r, s_t_2, s_t_1_, a_, r_, s_t_2_, mu, var)
            ### TODO contrastive loss

            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            loss.backward()  # 将误差反向传播
            vae_optimizer.step()  # 更新参数
            writer.add_scalar('loss_all', loss.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('BCE', BCE.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('KLD', KLD.item(), (epoch+1)*i*args.batch_size_)

        path = args.save_logdir + '/' + args.log_time + '/models/vae_epoch_' + str(epoch) + '.pt'
        utils.create_folders_if_necessary(path)
        torch.save(vae, path)








