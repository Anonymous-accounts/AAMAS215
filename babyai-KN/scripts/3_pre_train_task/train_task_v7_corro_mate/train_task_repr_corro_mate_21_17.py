#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

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
# from babyai.models.encoder_att import *
from babyai.models.encoder_mate import *
from babyai.models.gat import *
from babyai.models.vae_task_encoder import *

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


args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v7_corro_mate/logs_models_v7/'
args.loader_path = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/data_3_task_new.npy'
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
    # import pdb
    # pdb.set_trace()
    ### other tasks as negative pairs
    task_num = train_samples_dataset.shape[0]
    task_id = np.random.randint(0, task_num)
    query_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    key_index = np.random.randint(0, train_samples_dataset.shape[1], size=(batch_size))
    # train_samples_dataset ::: (3, 210000, 48)
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
    episode_length = 200
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
    episode_length = 200
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
    tasks = [0, 1, 2, 3]
    if random_flag:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch_random(npy_loader, tasks)
    else:
        obs_context, actions_context, rewards_context, next_obs_context, _ = sample_context_batch(npy_loader, tasks)
    #print(obs_context.shape)
    n_timesteps, n_tasks, _ = obs_context.shape
    _, _, _, encodings = vae.encoder(
            obs_context.reshape(n_timesteps*n_tasks, -1),
            actions_context.reshape(n_timesteps*n_tasks, -1),
            rewards_context.reshape(n_timesteps*n_tasks, -1)
            # next_obs_context.reshape(n_timesteps*n_tasks, -1)
        )
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

    # vgae = VGAE(args)
    # vgae.train()
    # vgae_optimizer = torch.optim.Adam(vgae.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # # define loss function 
    # criterion = nn.CrossEntropyLoss() 

    args.aggregator_hidden_size = 128
    args.task_embedding_size = 5
    args.obs_dim = 147
    args.rew_dim = 1
    args.normalize_z = True
    args.layer_norm = True
    args.encoder_lr = 0.0003
    args.contrastive_batch_size = 64
    args.n_negative_per_positive = 16
    args.reward_std = 0.5
    args.infonce_temp = 0.1
    args.sizes = [args.obs_dim, args.n_actions, 1, args.obs_dim, 1]
    args.pos_neg_gap = 0.05
    args.activation = 'relu'
    args.hiddens = [256, 128, 64, 32]
    args.obs_loss_coef, args.rew_loss_coef, args.kl_loss_coef, args.cons_loss_coef = 1.0, 1.0, 0.003, 0.5
    
    vae = VAE(args.task_embedding_size, args.obs_dim, args.n_actions, args.rew_dim, args.n_agents, args.hiddens, args.activation, args.layer_norm)

    encoder_optimizer = torch.optim.Adam(vae.parameters(), lr=args.encoder_lr)

    # gat_adj = encoder.contruct_edges(args.contrastive_batch_size, args.n_negative_per_positive, args.pos_neg_gap)


    # save arguments
    argsDict = args.__dict__
    with open(args.save_logdir + '/' + args.log_time + '/arguements.txt', 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')


    print('begin to train the task representation')
    # path_root = '/home/username/data/BABYAI/best_data'
    # loader_path = read_data_from_npy_save_loader_wo_pad(best_data_path)  # done
    ### read data_from_dataloader
    # loader_path = args.save_logdir + '/train_loader'
    
    # train_loader = torch.load(loader_path)
    npy_loader = np.load(args.loader_path)

    for epoch in tqdm(range(args.n_epoch)): # self.training_mode(True)
        vae.train(True)
        # args.actual_max_eps_len = 46
        # args.feature_length = 302
        # train_x = images.view(args.batch_size_, args.actual_max_eps_len, -1) # 64 1 302
        
        queries, keys, negs = sample_pairs_than_others(npy_loader, args.contrastive_batch_size, args.sizes) # 64
        
        obs_q, actions_q, rewards_q, next_obs_q, label_q = queries
        obs_k, actions_k, rewards_k, next_obs_k, label_k = keys
        obs_neg, actions_neg, rewards_neg, next_obs_neg, label_neg = negs
        

        b_dot_N = args.contrastive_batch_size * args.n_negative_per_positive # 64 * 16
        task_emb_q, mu_q, log_var_q, z_q = vae.encoder(obs_q, actions_q, rewards_q)
        task_emb_k, mu_k, log_var_k, z_k = vae.encoder(obs_k, actions_k, rewards_k)
        task_emb_neg, mu_neg, log_var_neg, z_neg = vae.encoder(obs_neg, actions_neg, rewards_neg)
        task_emb_q_train, task_emb_k_train, task_emb_neg_train = z_q, z_k, z_neg

        # print(torch.mean(mu_q), torch.mean(mu_k), torch.mean(mu_neg))
        # print(torch.mean(log_var_q), torch.mean(log_var_k), torch.mean(log_var_neg))


        # contrastive_loss_ = contrastive_loss(task_emb_q, task_emb_k, task_emb_neg) # 64,5   64,5   64,16,5  --> scalar
        contrastive_loss_ = contrastive_loss(task_emb_q_train, task_emb_k_train, task_emb_neg_train) # 64,5   64,5   64,16,5  --> scalar
        
        kl_loss = -0.5 * torch.sum(1 + log_var_q - mu_q.pow(2) - log_var_q.exp())

        pred_obss, pred_rews = vae.decoder(task_emb_q_train, obs_q, actions_q)
        obs_loss = (next_obs_q - pred_obss).pow(2).mean()
        rew_loss = (rewards_k - pred_rews).pow(2).mean()

        # print(contrastive_loss)
        loss =  args.obs_loss_coef * obs_loss + args.rew_loss_coef * rew_loss + args.kl_loss_coef * kl_loss + args.cons_loss_coef * contrastive_loss_
        print('obs_loss, obs_loss, kl_loss, contrastive_loss_:::', obs_loss.item(), obs_loss.item(), kl_loss.item(), contrastive_loss_.item())
        # print('contrastive_loss_::', contrastive_loss_.item())

        encoder_optimizer.zero_grad()
        contrastive_loss_.backward()
        encoder_optimizer.step()

        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('obs_loss', obs_loss.item(), epoch)
        writer.add_scalar('rew_loss', rew_loss.item(), epoch)
        writer.add_scalar('kl_loss', kl_loss.item(), epoch)
        writer.add_scalar('contrastive_loss', contrastive_loss_.item(), epoch)

        if epoch % 20000 == 0:
            path = args.save_logdir + '/' + args.log_time + '/models/vae_epoch_' + str(epoch) + '.pt'
            utils.create_folders_if_necessary(path)
            torch.save(vae, path)
        if epoch % 10000 == 0: # 1000000
            random_flag = False
            save_path_ = args.save_logdir + '/' + args.log_time + '/vis_z_eps_200/'
            utils.create_folders_if_necessary(save_path_)
            vis_sample_embeddings(random_flag, npy_loader, os.path.join(save_path_, "train_fig{0}.png".format(epoch)))

            random_flag = True
            save_path_ = args.save_logdir + '/' + args.log_time + '/vis_zrand_200/'
            utils.create_folders_if_necessary(save_path_)
            vis_sample_embeddings(random_flag, npy_loader, os.path.join(save_path_, "train_fig{0}.png".format(epoch)))









