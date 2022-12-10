#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""
import time
import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from info_nce import InfoNCE

import babyai
import babyai.utils as utils
from babyai.arguments import ArgumentParser
# from babyai.models.m_1_vae import _autoencoder
# from babyai.models.vgae import VGAE, data_transfer
from babyai.models.encoder import *
from babyai.models.m_enc_1_dec import m_encoder_1_decoder

from utils.recursive_read import return_npy_events_list
# from .utils.mmd import mmd_
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from sklearn import manifold

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")

parser.add_argument("--KL_weight", type=float, default=0.001)
parser.add_argument("--BCE_weight", type=float, default=5.0)
parser.add_argument("--infonce_weight", type=float, default=1000.0)
parser.add_argument("--disentanglement_penalty", type=float, default=10.0)

# parser.add_argument("--contra_loss", type=str, default='contra_loss_')
parser.add_argument("--contra_loss", type=str, default='')
parser.add_argument("--MMD", type=str, default='_MMD_')
parser.add_argument("--MAX_EPS_LEN", type=int, default=16) # decide the length of padding
parser.add_argument("--batch_size_", type=int, default=64)
parser.add_argument("--n_epoch", type=int, default=10000000)

args = parser.parse_args()
args.tb = True

log_name_parts = {
    'env': 'train_loader_bs',
    'batch_size_': args.batch_size_,
    'algo': 'corro',
    'sample': 'others_neg',
    }
args.default_log_name = "{env}_{batch_size_}_{algo}_{sample}_".format(**log_name_parts)


args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_2_tasks_new/'

# #args.loader_path = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/data_3_task_new_Open_PutNextLocal_UnblockPickup.npy'
args.loader_path = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/data_4_task_new_Open_Goto20000_UnblockPickup_PutNextLocal.npy'

args.actual_max_eps_len = 1  # transition, here is 1, original 576
args.feature_length = 302

cur_time = datetime.now() + timedelta(hours=12)
args.log_time = cur_time.strftime("[%m-%d]%H.%M.%S")+ args.default_log_name

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


def sample_positive_pairs(train_samples_dataset, batch_size, sizes, trainset=True):
    # self.train_samples_dataset.shape : (1, 210000, 48)
    tasks_index = np.random.randint(0, train_samples_dataset.shape[0], size=(batch_size))
    query_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    key_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    # train_samples_dataset ::: (3, 210000, 48)
    query = train_samples_dataset[tasks_index, query_index] # (batchsize, 48)
    key = train_samples_dataset[tasks_index, key_index] # (batchsize, 48)
    
    query = ptu.FloatTensor(query)
    key = ptu.FloatTensor(key)
    query = torch.split(query, sizes, dim=1)
    key = torch.split(key, sizes, dim=1)
    assert len(query)==5 and len(key)==5

    return query, key


def sample_pairs_than_others(train_samples_dataset, batch_size, sizes):
    ### other tasks as negative pairs
    task_num = train_samples_dataset.shape[0]
    task_id = np.random.randint(0, task_num)
    query_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    key_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    # train_samples_dataset ::: (3, 210000, 303)
    query = train_samples_dataset[task_id, query_index] # (batchsize, 48)
    key = train_samples_dataset[task_id, key_index] # (batchsize, 48)
    
    query = ptu.FloatTensor(query)
    key = ptu.FloatTensor(key)
    query = torch.split(query, sizes, dim=1)
    key = torch.split(key, sizes, dim=1)
    # assert len(query)==5 and len(key)==5

    ##### negtive pairs  # n_negative_per_positive
    neg_task_id = np.delete(np.arange(task_num), task_id)
    neg_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size * args.n_negative_per_positive)) # (1024,)
    neg_task_id = neg_task_id.repeat(neg_index.shape[0]//len(neg_task_id))
    for _ in range(batch_size * args.n_negative_per_positive - len(neg_task_id)):
        neg_task_id = np.append(neg_task_id, neg_task_id[-1])
    # neg = train_samples_dataset[neg_task_id][:, neg_index] # (64, 64*8, 303)
    # 
    # import pdb
    # pdb.set_trace()
    neg = train_samples_dataset[neg_task_id, neg_index] # 
    neg = neg.reshape(batch_size, -1, neg.shape[-1]) # (64, 16, 303)
    neg = ptu.FloatTensor(neg)
    neg = torch.split(neg, sizes, dim=-1)
    assert len(query)==5 and len(key)==5 and len(neg)==5

    return query, key, neg

def create_negatives(state, action, num_negatives, next_state=None, reward=None): # reward_randomize
    next_obs = next_state.unsqueeze(1).expand(-1, num_negatives, -1) # 64,20 --> 64,16,20
    rewards = reward.reshape(-1,1,1).expand(-1, num_negatives, -1).clone() # 64,1 --> 64,16,1
    sp = rewards.shape
    noise = torch.normal(mean=0., std=args.reward_std, size=sp).to(ptu.device) # 0.5
    rewards += noise
    return rewards, next_obs

def contrastive_loss(q, k, neg):
    N = neg.shape[1]
    b = q.shape[0]
    l_pos = torch.bmm(q.view(b, 1, -1), k.view(b, -1, 1)) # (b,1,1)
    l_neg = torch.bmm(q.view(b, 1, -1), neg.transpose(1,2)) # (b,1,N)
    logits = torch.cat([l_pos.view(b, 1), l_neg.view(b, N)], dim=1)
    
    labels = torch.zeros(b, dtype=torch.long)
    labels = labels.to(ptu.device)
    cross_entropy_loss = nn.CrossEntropyLoss()
    _loss = cross_entropy_loss(logits/args.infonce_temp, labels)
    #print(logits, labels, loss)
    return _loss

def sample_context_batch(contextset, tasks):
    # contextset :: [3, 40w, 302]
    # contextset[0][0].shape : (200, 1050, 20)
    #i_episodes = np.random.choice(contextset[0][0].shape[1], self.args.num_context_trajs)
    ## TO-DO: trajectorey length not fixed
    num_context_trajs = 1
    episode_length = 5000
    context = []
    for i in tasks: # [0,1,2]
        i_episodes = np.random.choice(contextset.shape[1]-episode_length, num_context_trajs)[0] # should be randomized at every task ,, Select the starting time
        context_i = contextset[i][i_episodes:i_episodes+episode_length]  # 200, 303
        context.append(context_i)
    context = np.array(context).transpose(1, 0, 2)  # (200, 3, 303)
    split_tuple = torch.split(ptu.FloatTensor(context), args.sizes, dim=-1)
    ret = [i for i in split_tuple]
    # [torch.Size([200, 3, 20]), torch.Size([200, 3, 6]),torch.Size([200, 3, 1]),torch.Size([200, 3, 20]), torch.Size([200, 3, 1])]
    return ret

def sample_context_batch_random(contextset, tasks):
    num_context_trajs = 1
    episode_length = 5000
    context = []
    for i in tasks: # [0,1,2]
        i_episodes = np.random.choice(contextset.shape[1], episode_length) # should be randomized at every task ,, Select the starting time
        context_i = contextset[i][i_episodes]  # 200, 303
        context.append(context_i)
    context = np.array(context).transpose(1, 0, 2)  # (200, 3, 303)
    split_tuple = torch.split(ptu.FloatTensor(context), args.sizes, dim=-1)
    ret = [i for i in split_tuple]
    # [torch.Size([200, 3, 20]), torch.Size([200, 3, 6]),torch.Size([200, 3, 1]),torch.Size([200, 3, 20]), torch.Size([200, 3, 1])]
    return ret

def vis_sample_embeddings(random_flag, npy_loader, save_path):
    # goals = self.goals if trainset else self.eval_goals
    x, y = [], []
    tasks = [0, 1]
    if random_flag:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch_random(npy_loader, tasks)
    else:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch(npy_loader, tasks)
    #print(obs_context.shape)
    n_timesteps, n_tasks, _ = obs_context.shape  ### 200.4
    encodings = torch.stack([m_encoder_1_decoder_.encoders[i](obs_context[:,i], actions_context[:,i], rewards_context[:,i], next_obs_context[:,i]) for i in tasks])
    encodings = encodings.transpose(1, 0) # 200.4.5
    # encodings = encoder(
    #         obs_context.reshape(n_timesteps*n_tasks, -1),
    #         actions_context.reshape(n_timesteps*n_tasks, -1),
    #         rewards_context.reshape(n_timesteps*n_tasks, -1),
    #         next_obs_context.reshape(n_timesteps*n_tasks, -1)
    #     )
    encodings = encodings.reshape(n_timesteps, n_tasks, -1).cpu().detach().numpy()
    obs_context, actions_context, rewards_context, next_obs_context = \
        obs_context.cpu().detach().numpy(), actions_context.cpu().detach().numpy(), \
        rewards_context.cpu().detach().numpy(), next_obs_context.cpu().detach().numpy()
    
    for i, t in enumerate(tasks):
        for j in range(obs_context.shape[0]):
            x.append(encodings[j,i])
            y.append(i)
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(np.asarray(x))

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    data = (X_tsne - x_min) / (x_max - x_min)

    colors = plt.cm.rainbow(np.linspace(0,1,len(tasks)))

    plt.cla()
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(y[i]),
                color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)





def vis_sample_embeddings_2(random_flag, npy_loader, save_path):
    
    # goals = self.goals if trainset else self.eval_goals
    x, y = [], []
    tasks = [0, 1]
    if random_flag:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch_random(npy_loader, tasks)
    else:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch(npy_loader, tasks)
    #print(obs_context.shape)
    n_timesteps, n_tasks, _ = obs_context.shape  ### 200.4
    encodings = torch.stack([m_encoder_1_decoder_.encoders[i](obs_context[:,i], actions_context[:,i], rewards_context[:,i], next_obs_context[:,i]) for i in tasks])
    encodings = encodings.transpose(1, 0) # 200.4.5
    # encodings = encoder(
    #         obs_context.reshape(n_timesteps*n_tasks, -1),
    #         actions_context.reshape(n_timesteps*n_tasks, -1),
    #         rewards_context.reshape(n_timesteps*n_tasks, -1),
    #         next_obs_context.reshape(n_timesteps*n_tasks, -1)
    #     )
    encodings = encodings.reshape(n_timesteps, n_tasks, -1).cpu().detach().numpy()
    obs_context, actions_context, rewards_context, next_obs_context = \
        obs_context.cpu().detach().numpy(), actions_context.cpu().detach().numpy(), \
        rewards_context.cpu().detach().numpy(), next_obs_context.cpu().detach().numpy()
    
    for i, t in enumerate(tasks):
        for j in range(obs_context.shape[0]):
            x.append(encodings[j,i])
            y.append(i)
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(np.asarray(x))

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    data = (X_tsne - x_min) / (x_max - x_min)

    save_path_cur = os.path.join(save_path, "test_fig_"+ "Set2_5000" +".png")
    import brewer2mpl
    # brewer2mpl.get_map args: set name set type number of colors
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    colors = bmap.mpl_colors

    plt.cla()
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=colors[y[i]], marker='o')
        # plt.text(data[i, 0], data[i, 1], str(y[i]),
        #         color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
        #         fontdict={'weight': 'bold', 'size': 9})
    plt.savefig(save_path_cur)

    
    ## https://medium.com/ken-m-lai/%E8%AE%A9matplotlib%E9%85%8D%E8%89%B2%E6%96%B9%E6%A1%88%E6%9B%B4%E5%8A%A0%E7%BE%8E%E8%A7%82%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F-3f5a91783aba
    
    # set_qua = ['Set3', 'Set2', 'Set1', 'Pastel2', 'Paired', 'Dark2', 'Accent']
    # for i in set_qua:
    #     save_path_cur = os.path.join(save_path, "test_fig_"+ i +".png")
    #     import brewer2mpl
    #     # brewer2mpl.get_map args: set name set type number of colors
    #     bmap = brewer2mpl.get_map(i, 'qualitative', 7)
    #     colors = bmap.mpl_colors

    #     plt.cla()
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     for i in range(data.shape[0]):
    #         plt.scatter(data[i, 0], data[i, 1], color=colors[y[i]], marker='o')
    #         # plt.text(data[i, 0], data[i, 1], str(y[i]),
    #         #         color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
    #         #         fontdict={'weight': 'bold', 'size': 9})
    #     plt.savefig(save_path_cur)


    # set_seq = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # for i in set_seq:
    #     save_path_cur = os.path.join(save_path, "test_fig_"+ i +".png")
    #     import brewer2mpl
    #     # brewer2mpl.get_map args: set name set type number of colors
    #     bmap = brewer2mpl.get_map(i, 'sequential', 7)
    #     colors = bmap.mpl_colors

    #     plt.cla()
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     for i in range(data.shape[0]):
    #         plt.scatter(data[i, 0], data[i, 1], color=colors[y[i]], marker='o')
    #         # plt.text(data[i, 0], data[i, 1], str(y[i]),
    #         #         color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
    #         #         fontdict={'weight': 'bold', 'size': 9})
    #     plt.savefig(save_path_cur)


    # set_div = ['BrBG', 'PRGn', 'PiYG', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral']
    # for i in set_div:
    #     save_path_cur = os.path.join(save_path, "test_fig_"+ i +".png")
    #     import brewer2mpl
    #     # brewer2mpl.get_map args: set name set type number of colors
    #     bmap = brewer2mpl.get_map(i, 'diverging', 7)
    #     colors = bmap.mpl_colors

    #     plt.cla()
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     for i in range(data.shape[0]):
    #         plt.scatter(data[i, 0], data[i, 1], color=colors[y[i]], marker='o')
    #         # plt.text(data[i, 0], data[i, 1], str(y[i]),
    #         #         color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
    #         #         fontdict={'weight': 'bold', 'size': 9})
    #     plt.savefig(save_path_cur)


    # set_mis = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
    #         'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
    # for i in set_mis:
    #     save_path_cur = os.path.join(save_path, "test_fig_"+ i +".png")
    #     import brewer2mpl
    #     # brewer2mpl.get_map args: set name set type number of colors
    #     bmap = brewer2mpl.get_map(i, 'diverging', 7)
    #     colors = bmap.mpl_colors

    #     plt.cla()
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     for i in range(data.shape[0]):
    #         plt.scatter(data[i, 0], data[i, 1], color=colors[y[i]], marker='o')
    #         # plt.text(data[i, 0], data[i, 1], str(y[i]),
    #         #         color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
    #         #         fontdict={'weight': 'bold', 'size': 9})
    #     plt.savefig(save_path_cur)





if __name__ == '__main__':
    print('initialize the model...')
    args.latent_dim = 147
    args.obs_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 64

    args.input_dim = 302 # args.latent_dim*2 + args.n_agents + args.n_actions
    args.n_agents = 1
    args.n_actions = 7
    args.hidden1_dim = 128
    args.hidden2_dim = 128

    args.obs_dim = args.hidden2_dim
    args.hidden_size = 64
    args.layer_N = 2
    args.out_dim = 3
    
    args.use_bn = True

    args.aggregator_hidden_size = 64
    args.task_embedding_size = 5
    args.obs_dim = 147
    args.normalize_z = True
    args.encoder_lr = 0.0003
    args.contrastive_batch_size = 64
    args.n_negative_per_positive = 16
    args.reward_std = 0.5
    args.infonce_temp = 0.1
    args.sizes = [args.obs_dim, args.n_actions, 1, args.obs_dim, 1]
    args.hidden_size_lst = [256, 256, 128, 128, 32]
    args.num_tasks = 2

    args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_2_tasks_new/'

    #args.loader_path = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/data_3_task_new_Open_PutNextLocal_UnblockPickup.npy'
    args.loader_path = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/data_4_task_new_Open_Goto20000_UnblockPickup_PutNextLocal.npy'


    # save arguments
    # argsDict = args.__dict__
    # with open(args.save_logdir + '/' + args.log_time + '/arguements.txt', 'w') as f:
    #     for eachArg, value in argsDict.items():
    #         f.writelines(eachArg + ' : ' + str(value) + '\n')
            
    # m_encoder_1_decoder_ = m_encoder_1_decoder(args)
    # args.log_time = '[09-07]11.26.22train_loader_bs_64_corro_others_neg_/'
    # args.model_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_4_tasks_new/[09-07]11.26.22train_loader_bs_64_corro_others_neg_/models/encoder_epoch_740000.pt'
    
    # args.log_time = 'id013_[09-11]16.31.50train_loader_bs_64_corro_others_neg_'
    # args.model_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_3_tasks_new/id013_[09-11]16.31.50train_loader_bs_64_corro_others_neg_/models/encoder_epoch_0.pt'
    
    args.log_time = '[09-09]19.53.24train_loader_bs_64_corro_others_neg_'
    args.model_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_2_tasks_new/[09-09]19.53.24train_loader_bs_64_corro_others_neg_/models/encoder_epoch_840000.pt'
    
    
    m_encoder_1_decoder_ = torch.load(args.model_path)
    print('begin to load the model for the task representation')
    
    npy_loader = np.load(args.loader_path)
    #npy_loader = npy_loader[:3]
    print('npy_loader::::::::::::::::::', npy_loader.shape)

    save_path_ = args.save_logdir + '/' + args.log_time + '/vis_z_plt/'
    utils.create_folders_if_necessary(save_path_)
    vis_sample_embeddings_2(False, npy_loader, save_path_)
    
        








