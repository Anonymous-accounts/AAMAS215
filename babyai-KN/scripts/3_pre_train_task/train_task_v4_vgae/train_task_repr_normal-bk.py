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
from babyai.models.m_1_vae import _autoencoder

from utils.recursive_read import return_npy_events_list
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
# Figure out what's the problem

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")

parser.add_argument("--KL_weight", type=float, default=1)
parser.add_argument("--infonce_weight", type=float, default=1000)
parser.add_argument("--contra_loss", type=str, default='')
parser.add_argument("--MAX_EPS_LEN", type=int, default=64) # decide the length of padding
parser.add_argument("--batch_size_", type=int, default=64)
parser.add_argument("--n_epoch", type=int, default=50)

args = parser.parse_args()
args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/pre_train_task/train_task_v4_vgae/logs_models_v4/'

args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/pre_train_task/datasets_episode/'
best_data_path = args.train_loader_save_logdir + 'best_data/'
loader_path = args.train_loader_save_logdir + 'train_loader_eps_bs' + str(args.batch_size_)

cur_time = datetime.now() + timedelta(hours=12)
args.log_time = cur_time.strftime("[%m-%d]%H.%M.%S") + 'KL_mul_' + str(args.KL_weight) + args.contra_loss
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


def padding_eps_data_to_max(data_x, begin, end): # 0, 63
    length = end - begin + 1
    pad_length = args.MAX_EPS_LEN - length

    preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][0]).image) for i in range(begin, end+1)]
    data_x_1 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_2_ = torch.tensor([data_x[i][1] for i in range(begin, end+1)]) # action
    data_x_2 = list_onehot(data_x_2_, args.n_actions)
    data_x_3 = torch.tensor([data_x[i][2] for i in range(begin, end+1)]).float()  # rewards
    preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][4]).image) for i in range(begin, end+1)]
    data_x_4 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_1234 = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)

    data_all_length = data_x_1234.shape[1]
    data_x_1234_pad = torch.nn.functional.pad(data_x_1234, (0,0,0,pad_length), mode='constant')
    return data_x_1234_pad

def read_data_from_npy_save_loader(path):

    npy_lists = return_npy_events_list(path)
    index = 0
    data_X = []
    data_Y = []
    for i in npy_lists:
        # import pdb
        index += 1
        data_x = np.load(i, allow_pickle=True) # 加载npy数据
        data_eps_pad = []
        for j in range(data_x.shape[0]):
            anchor_1, anchor_2 = 0, 0
            if data_x[j][5] == 0:
                anchor_1 = j
                for j_2 in range(j+1, data_x.shape[0]):
                    if data_x[j_2][5] == 0:
                        anchor_2 = j_2 - 1
                        break
                if anchor_2 == 0:
                    break
                j = j_2 
                data_padding = padding_eps_data_to_max(data_x, anchor_1, anchor_2)
                data_eps_pad.append(data_padding)
                # print(anchor_1, anchor_2)
        
        data_eps_pad = torch.stack(data_eps_pad) # 9999,64*302
        data_eps_pad = data_eps_pad.view(data_eps_pad.shape[0], -1)
        data_y = (torch.ones(data_eps_pad.shape[0])*index).int()
        data_X.append(data_eps_pad)
        data_Y.append(data_y)
        print(i)
    train_x = torch.cat(data_X) # torch.Size([767293, 147])
    train_y = torch.cat(data_Y) # torch.Size([767293])
    Train_DS = TensorDataset(train_x, train_y)
    Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=args.batch_size_)
    torch.save(Train_DL, args.train_loader_save_logdir + '/train_loader_eps_bs' + str(args.batch_size_))
    print('load and save done....')
    return args.train_loader_save_logdir + '/train_loader_eps_bs' + str(args.batch_size_)

def data_normal_2d(orign_data, dim="col"):
    """
    	针对于2维tensor归一化
    	可指定维度进行归一化，默认为行归一化
    """
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    # norm_data = torch.divide(torch.sub(orign_data,d_min), dst)
    norm_data = torch.sub(orign_data,d_min)/ dst
    return norm_data

if __name__ == '__main__':
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 64
    vae = _autoencoder(args)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print('begin to train the task representation')
    # path_root = '/home/username/data/BABYAI/best_data'
    # loader_path = read_data_from_npy_save_loader(best_data_path)  # done
    ### read data_from_dataloader
    # loader_path = args.save_logdir + '/train_loader'
    # break
    # print(a_2)
    train_loader = torch.load(loader_path)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(train_loader):  # 100*32
            print("epoch, minibatch of traning vae ------", epoch, i)
            s_t_1, a, r, s_t_2 = images[:, :147], images[:, 147:147+args.n_actions], images[:, [147++args.n_actions]], images[:, -147:] # s,a,r,s'
            s_t_1 = data_normal_2d(s_t_1)
            s_t_2 = data_normal_2d(s_t_2)
            # import pdb
            # pdb.set_trace()
            s_t_1_, a_, r_, s_t_2_, mu, var, z = vae(s_t_1, a, r, s_t_2, labels)
            s_t_1_ = data_normal_2d(s_t_1_)
            s_t_2_ = data_normal_2d(s_t_2_)
            # 32.100, 32.7, 32.1, 32.100, 32.100, 32.100
            loss, BCE, KLD = vae.backward(s_t_1, a, r, s_t_2, s_t_1_, a_, r_, s_t_2_, mu, var)
            ### TODO contrastive loss
            # import pdb
            # pdb.set_trace()
            if args.contra_loss == '':
                loss_all = BCE + args.KL_weight * KLD
            else:
                loss_infonce = InfoNCE(negative_mode='unpaired')
                loss_infonce_lst = []
                query = z
                for j in range(5):  ## 5  tasks
                    index_positive_keys = torch.where(labels == j)[0]
                    index_negative_keys = torch.where(labels != j)[0]
                    if len(index_positive_keys) > 0:
                        positive_keys = torch.index_select(z, 0, index_positive_keys)
                        negative_keys = torch.index_select(z, 0, index_negative_keys)
                        query = positive_keys
                        loss_ = loss_infonce(query, positive_keys, negative_keys)
                        loss_infonce_lst.append(loss_)
                loss_infonce_ = torch.stack(loss_infonce_lst).mean()
                writer.add_scalar('loss_infonce_', loss_infonce_.item(), (epoch+1)*i*args.batch_size_)
                loss_all = args.infonce_weight * loss_infonce_ + BCE + args.KL_weight * KLD

            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_all.backward()  # 将误差反向传播
            vae_optimizer.step()  # 更新参数
            writer.add_scalar('loss_all', loss_all.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('BCE', BCE.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('KLD', KLD.item(), (epoch+1)*i*args.batch_size_)
            

        path = args.save_logdir + '/' + args.log_time + '/models/vae_epoch_' + str(epoch) + '.pt'
        utils.create_folders_if_necessary(path)
        torch.save(vae, path)








