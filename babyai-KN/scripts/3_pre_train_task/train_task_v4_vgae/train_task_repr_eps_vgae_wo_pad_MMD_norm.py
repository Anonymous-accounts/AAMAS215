#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import numpy as np
import torch
from tqdm import tqdm
from info_nce import InfoNCE

import babyai
import babyai.utils as utils
from babyai.arguments import ArgumentParser
# from babyai.models.m_1_vae import _autoencoder
from babyai.models.vgae import VGAE, data_transfer

from utils.recursive_read import return_npy_events_list
from utils.mmd import mmd_
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
# Figure out what's the problem

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")

parser.add_argument("--KL_weight", type=float, default=0.01)
parser.add_argument("--BCE_weight", type=float, default=5.0)
parser.add_argument("--infonce_weight", type=float, default=1000.0)
parser.add_argument("--disentanglement_penalty", type=float, default=10.0)

# parser.add_argument("--contra_loss", type=str, default='contra_loss_')
parser.add_argument("--contra_loss", type=str, default='')
parser.add_argument("--MMD", type=str, default='_MMD_')
parser.add_argument("--MAX_EPS_LEN", type=int, default=16) # decide the length of padding
parser.add_argument("--batch_size_", type=int, default=64)
parser.add_argument("--n_epoch", type=int, default=50)

args = parser.parse_args()
args.tb = True
args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/train_task_v4_vgae/logs_models_v4_differ4_norm/'

args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/datasets_episode/'
best_data_path = args.train_loader_save_logdir + 'best_data/'
loader_path = args.train_loader_save_logdir + 'train_loader_differ4_eps_bs' + str(args.batch_size_) + '_wo_pad'
args.actual_max_eps_len = 576
args.feature_length = 302

cur_time = datetime.now() + timedelta(hours=12)
if args.contra_loss != '':
    args.log_time = cur_time.strftime("[%m-%d]%H.%M.%S")+ '_bs_' + str(args.batch_size_) + '_wo_pad' + '_BCE_' + str(args.BCE_weight) + '_KL_' + str(args.KL_weight) \
    + args.contra_loss + str(args.infonce_weight) + args.MMD + str(args.disentanglement_penalty)
else:
    args.log_time = cur_time.strftime("[%m-%d]%H.%M.%S")+ '_bs_' + str(args.batch_size_) + '_wo_pad' + '_BCE_' + str(args.BCE_weight) + '_KL_' + str(args.KL_weight)

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


def generate_seq_adj(x_, MAX_EPS_LEN):
    pad_zeros_first_index = 0
    for i in range(MAX_EPS_LEN-1, 0, -1):
        if(np.all(x_[i].numpy() == 0) == False):
            pad_zeros_first_index = i + 1
            break
    adj_zeros_numpy = np.zeros((pad_zeros_first_index, pad_zeros_first_index))
    for i in range(0, pad_zeros_first_index-1):
        adj_zeros_numpy[i][i+1] = 1
    return adj_zeros_numpy, pad_zeros_first_index

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

if __name__ == '__main__':
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 64

    args.input_dim = 302 # args.latent_dim*2 + args.n_agents + args.n_actions
    args.n_agents = 1
    args.n_actions = 7
    args.hidden1_dim = 64
    args.hidden2_dim = 32
    
    args.use_bn = True
    vgae = VGAE(args)
    vgae.train()

    vgae_optimizer = torch.optim.Adam(vgae.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print('begin to train the task representation')
    # path_root = '/home/username/data/BABYAI/best_data'
    # loader_path = read_data_from_npy_save_loader_wo_pad(best_data_path)  # done
    ### read data_from_dataloader
    # loader_path = args.save_logdir + '/train_loader'
    
    train_loader = torch.load(loader_path)

    for epoch in tqdm(range(args.n_epoch)):
        count_0 = 0
        for i, (images, labels) in tqdm(enumerate(train_loader)):  # 100*32
            if images.view(-1).shape[0] != args.batch_size_ * args.actual_max_eps_len * args.feature_length:
                continue
            print(i)
            # args.actual_max_eps_len = 46
            # args.feature_length = 302
            train_x = images.view(args.batch_size_, args.actual_max_eps_len, -1) # 64 46 302
            log_lik_lst, kl_divergence_lst = [], []
            z_lst = []
            for index in range(train_x.shape[0]):
                x_ = train_x[index] ### 64, 302 可以normalize
                adj_, pad_zeros_first_index = generate_seq_adj(x_, args.actual_max_eps_len)
                if x_[:pad_zeros_first_index].shape[0] == 0:
                    count_0 += 1
                    labels = del_tensor_ele(labels, index)
                    continue
            

                x_clip = x_[:pad_zeros_first_index]
                # x: [num_of_point, dim]
                if vgae.use_bn:
                    x_clip = x_clip.unsqueeze(2)
                    x_clip = x_clip.transpose(0, 2)             # # x: [1, dim， num_of_point]
                    x_clip = vgae.batch_norm(x_clip)
                    x_clip = x_clip.transpose(0, 2).squeeze()   # # x: [num_of_point, dim]
                
                # adj_norm, adj_label, features, norm, weight_tensor = data_transfer(adj_, x_[:pad_zeros_first_index])
                adj_norm, adj_label, norm, weight_tensor = data_transfer(adj_)
                z, A_pred = vgae(x_clip, adj_norm) 
                # z, A_pred = vgae(features, adj_norm) 
                log_lik, kl_divergence = vgae.backward(A_pred, adj_label, norm, weight_tensor)

                log_lik_lst.append(log_lik)
                kl_divergence_lst.append(kl_divergence)
                z_lst.append(torch.mean(z, 0, keepdim=True))
            
            
            BCE = torch.tensor(log_lik_lst).mean()
            KLD = torch.tensor(kl_divergence_lst).mean()
            z = torch.cat(z_lst, dim=0)
            writer.add_scalar('count_0', count_0, (epoch+1)*i*args.batch_size_)
            writer.add_scalar('BCE', BCE.item(), (epoch+1)*i*args.batch_size_)
            writer.add_scalar('KLD', KLD.item(), (epoch+1)*i*args.batch_size_)
            if args.contra_loss == '':
                loss_all = args.BCE_weight * BCE + args.KL_weight * KLD
                writer.add_scalar('loss_all', loss_all.item(), (epoch+1)*i*args.batch_size_)
            else:
                # import pdb
                # pdb.set_trace()
                if z.shape[0] == labels.shape[0]:
                    gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)]) # torch.Size([13])
                    loss_mmd_lst = []
                    loss_infonce = InfoNCE(negative_mode='unpaired')
                    loss_infonce_lst = []
                    query = z
                    for j in range(3):  ## 4  tasks
                        # if i == 45:
                        #     import pdb
                        #     pdb.set_trace()
                        index_positive_keys = torch.where(labels == j)[0]
                        index_negative_keys = torch.where(labels != j)[0]
                        if len(index_positive_keys) > 0:
                            positive_keys = torch.index_select(z, 0, index_positive_keys)
                            negative_keys = torch.index_select(z, 0, index_negative_keys)
                            query = positive_keys
                            loss_ = loss_infonce(query, positive_keys, negative_keys)
                            loss_infonce_lst.append(loss_)
                            loss_mmd = mmd_(positive_keys, negative_keys, gammas=gammas) 
                            loss_mmd_lst.append(loss_mmd)
                    loss_infonce_ = torch.stack(loss_infonce_lst).mean()
                    loss_mmd_ = torch.stack(loss_mmd_lst).mean()
                    writer.add_scalar('loss_infonce_', loss_infonce_.item(), (epoch+1)*i*args.batch_size_)
                    writer.add_scalar('loss_mmd_', loss_mmd_.item(), (epoch+1)*i*args.batch_size_)
                    
                    loss_all = args.BCE_weight * BCE + args.KL_weight * KLD + \
                        args.infonce_weight * loss_infonce_ + args.disentanglement_penalty * loss_mmd_
                        


                    vgae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
                    loss_all.backward()  # 将误差反向传播
                    vgae_optimizer.step()  # 更新参数

                    writer.add_scalar('loss_all', loss_all.item(), (epoch+1)*i*args.batch_size_)
                    print('loss_all.item()', loss_all.item(), 'BCE.item()', BCE.item(), \
                    'KLD.item()', KLD.item(), 'loss_infonce_', loss_infonce_.item(), 'loss_mmd_.item()', loss_mmd_.item())
            
            ### TODO: 1. state normalization 2. contrastive loss 3. MMD loss
        print('the first dimension of feature shape: == 0:::', count_0)
        path = args.save_logdir + '/' + args.log_time + '/models/vae_epoch_' + str(epoch) + '.pt'
        utils.create_folders_if_necessary(path)
        torch.save(vgae, path)








