#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import numpy as np
import torch
from info_nce import InfoNCE

import babyai
import babyai.utils as utils
from babyai.arguments import ArgumentParser
from babyai.models.vae import _autoencoder

from utils.recursive_read import return_npy_events_list
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
# Figure out what's the problem


#### 区分 log
differ_logs = 'KL_mul_0.1_contra_loss_'


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()
args.save_logdir = './logs_models'
args.n_epoch = 30
args.batch_size_ = 32

cur_time = datetime.now() + timedelta(hours=0)
args.log_time = differ_logs + cur_time.strftime("[%m-%d]%H.%M.%S")
writer = SummaryWriter(logdir=args.save_logdir + '/' + args.log_time + "/logs")

utils.seed(args.seed)

obss_preprocessor = utils.ObssPreprocessor(args.save_logdir, None, None)


def trans_image(image):
    return torch.transpose(torch.transpose(image, 1, 3), 2, 3)

def list_onehot(actions: list, n: int) -> torch.Tensor:
    """
    列表动作值转 onehot
    actions: 动作列表
    n: 动作总个数
    """
    result = []
    for action in actions:
        result.append([int(k == action) for k in range(n)])
    result = torch.tensor(result, dtype=torch.float)
    return result

def read_data_from_npy_save_loader(path):

    npy_lists = return_npy_events_list(path)
    index = 0
    data_X = []
    data_Y = []
    for i in npy_lists:
        index += 1
        data_x = np.load(i) # 加载npy数据
        preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][0]).image) for i in range(data_x.shape[0])]
        data_x_1 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
        data_x_2_ = torch.tensor([data_x[i][1] for i in range(data_x.shape[0])]) # action
        data_x_2 = list_onehot(data_x_2_, args.n_actions)
        data_x_3 = torch.tensor([data_x[i][2] for i in range(data_x.shape[0])])
        preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][4]).image) for i in range(data_x.shape[0])]
        data_x_4 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
        data_x = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)
        data_y = (torch.ones(data_x.shape[0])*index).int()
        data_X.append(data_x)
        data_Y.append(data_y)
    train_x = torch.cat(data_X) # torch.Size([767293, 147])
    train_y = torch.cat(data_Y) # torch.Size([767293])
    Train_DS = TensorDataset(train_x, train_y)
    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=args.batch_size_)
    torch.save(Train_DL, args.save_logdir + '/train_loader')
    print('load and save done....')
    return args.save_logdir + '/train_loader'



if __name__ == '__main__':
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 128
    vae = _autoencoder(args)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print('begin to train the task representation')
    path_root = '/home/username/data/BABYAI/best_data'
    # loader_path = read_data_from_npy_save_loader(path_root)  # done
    ### read data_from_dataloader
    loader_path = args.save_logdir + '/train_loader'
    train_loader = torch.load(loader_path)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(train_loader):  # 100*32
            print("epoch, minibatch of traning vae ------", epoch, i)
            s_t_1, a, r, s_t_2 = images[:, :147], images[:, 147:147+args.n_actions], images[:, [147++args.n_actions]], images[:, -147:] # s,a,r,s'
            s_t_1_, a_, r_, s_t_2_, mu, var, z = vae(s_t_1, a, r, s_t_2)
            # 32.100, 32.7, 32.1, 32.100, 32.100, 32.100, 32.147
            loss, BCE, KLD = vae.backward(s_t_1, a, r, s_t_2, s_t_1_, a_, r_, s_t_2_, mu, var)
            loss_new = BCE + 0.08 * KLD
            ### TODO contrastive loss
            loss_infonce = InfoNCE(negative_mode='unpaired')
            loss_infonce_lst = []
            query = z
            for i in range(5): ## 5  tasks
                index_positive_keys = torch.where(labels == i)[0]
                index_negative_keys = torch.where(labels != i)[0]
                if len(index_positive_keys) > 0:
                    positive_keys = torch.index_select(z, 0, index_positive_keys)
                    negative_keys = torch.index_select(z, 0, index_negative_keys)
                    query = positive_keys
                    loss_ = loss_infonce(query, positive_keys, negative_keys)
                    loss_infonce_lst.append(loss_)
            for i in range(len(labels)):
                query = z[[i], :]
            loss_infonce_ = torch.stack(loss_infonce_lst).mean()
            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            (loss_new + loss_infonce_).backward()  # 将误差反向传播
            vae_optimizer.step()  # 更新参数
            writer.add_scalar('loss_old', loss.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('loss_new', loss_new.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('BCE', BCE.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('KLD', KLD.item(), (epoch+1)*i*args.batch_size_)

        path = args.save_logdir + '/' + args.log_time + '/models/vae_epoch_' + str(epoch) + '.pt'
        utils.create_folders_if_necessary(path)
        torch.save(vae, path)








