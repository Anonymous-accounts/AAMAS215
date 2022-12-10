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
from babyai.models.encoder_att import *
from babyai.models.gat import *

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


args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v6_corro_att/logs_models_v6_att/'

args.loader_path = '/data2/username_high/username/BABYAI/data_new/data_4_task.npy'

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
    encodings = encoder(
            obs_context.reshape(n_timesteps*n_tasks, -1),
            actions_context.reshape(n_timesteps*n_tasks, -1),
            rewards_context.reshape(n_timesteps*n_tasks, -1),
            next_obs_context.reshape(n_timesteps*n_tasks, -1)
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
    args.normalize_z = True
    args.encoder_lr = 0.0003
    args.contrastive_batch_size = 64
    args.n_negative_per_positive = 16
    args.reward_std = 0.5
    args.infonce_temp = 0.1
    args.sizes = [args.obs_dim, args.n_actions, 1, args.obs_dim, 1]
    args.pos_neg_gap = 0.05

    encoder = MLPEncoder(
                hidden_size=args.aggregator_hidden_size, # 256
                num_hidden_layers=2,
                task_embedding_size=args.task_embedding_size, # 5
                action_size=args.n_actions, # 7
                state_size=args.obs_dim, # 20
                reward_size=1,
                term_size=0, # encode (s,a,r,s') only
                normalize=args.normalize_z # True
        	).to(ptu.device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)

    gat_adj = encoder.contruct_edges(args.contrastive_batch_size, args.n_negative_per_positive, args.pos_neg_gap)


    print('begin to train the task representation')
    # path_root = '/home/username/data/BABYAI/best_data'
    # loader_path = read_data_from_npy_save_loader_wo_pad(best_data_path)  # done
    ### read data_from_dataloader
    # loader_path = args.save_logdir + '/train_loader'
    
    # train_loader = torch.load(loader_path)
    npy_loader = np.load(args.loader_path)

    for epoch in tqdm(range(args.n_epoch)): # self.training_mode(True)
        encoder.train(True)
        # args.actual_max_eps_len = 46
        # args.feature_length = 302
        # train_x = images.view(args.batch_size_, args.actual_max_eps_len, -1) # 64 1 302
        
        # queries, keys = sample_positive_pairs(npy_loader, args.contrastive_batch_size, args.sizes) # 64
        queries, keys, negs = sample_pairs_than_others(npy_loader, args.contrastive_batch_size, args.sizes) # 64
        
        obs_q, actions_q, rewards_q, next_obs_q, label_q = queries
        obs_k, actions_k, rewards_k, next_obs_k, label_k = keys
        obs_neg, actions_neg, rewards_neg, next_obs_neg, label_neg = negs

        # rewards_neg, next_obs_neg = create_negatives(obs_q, actions_q, args.n_negative_per_positive, next_state=next_obs_q, reward=rewards_q)

        # obs_neg = obs_q.unsqueeze(1).expand(-1, args.n_negative_per_positive, -1) # expand obs_q to (b, n_neg, dim), they share the same (s,a) 
        # actions_neg = actions_q.unsqueeze(1).expand(-1, args.n_negative_per_positive, -1)

        b_dot_N = args.contrastive_batch_size * args.n_negative_per_positive # 64 * 16
        q_z = encoder.forward(obs_q, actions_q, rewards_q, next_obs_q) # 64,5
        k_z = encoder.forward(obs_k, actions_k, rewards_k, next_obs_k)
        neg_z = encoder.forward(obs_neg.reshape(b_dot_N, -1), actions_neg.reshape(b_dot_N, -1), 
            rewards_neg.reshape(b_dot_N, -1), next_obs_neg.reshape(b_dot_N, -1)).view(args.contrastive_batch_size, args.n_negative_per_positive, -1)
        #import pdb
        #pdb.set_trace()
        xx = torch.cat((q_z, k_z, neg_z.reshape(neg_z.size(0) * neg_z.size(1), -1)), dim=0)
        yy = encoder.gat_model(xx, gat_adj)
        q_z, k_z = yy[:args.contrastive_batch_size], yy[args.contrastive_batch_size:args.contrastive_batch_size*2]
        neg_z = yy[args.contrastive_batch_size*2:].view(args.contrastive_batch_size, args.n_negative_per_positive, -1)

        contrastive_loss_ = contrastive_loss(q_z, k_z, neg_z) # 64,5   64,5   64,16,5  --> scalar
        # print(contrastive_loss)

        encoder_optimizer.zero_grad()
        contrastive_loss_.backward()
        encoder_optimizer.step()

        writer.add_scalar('contrastive_loss', contrastive_loss_.item(), epoch)

        if epoch % 20000 == 0:
            path = args.save_logdir + '/' + args.log_time + '/models/encoder_epoch_' + str(epoch) + '.pt'
            utils.create_folders_if_necessary(path)
            torch.save(encoder, path)
        if epoch % 10000 == 0: # 1000000
            random_flag = False
            save_path_ = args.save_logdir + '/' + args.log_time + '/vis_z_eps_200/'
            utils.create_folders_if_necessary(save_path_)
            vis_sample_embeddings(random_flag, npy_loader, os.path.join(save_path_, "train_fig{0}.png".format(epoch)))

            random_flag = True
            save_path_ = args.save_logdir + '/' + args.log_time + '/vis_zrand_200/'
            utils.create_folders_if_necessary(save_path_)
            vis_sample_embeddings(random_flag, npy_loader, os.path.join(save_path_, "train_fig{0}.png".format(epoch)))
        

            # log_lik_lst, kl_divergence_lst = [], []
            # z_lst = []
            # for index in range(train_x.shape[0]):
            #     x_ = train_x[index] ### 64, 302 可以normalize
            #     adj_, pad_zeros_first_index = generate_seq_adj(x_, args.actual_max_eps_len)
            #     if x_[:pad_zeros_first_index].shape[0] == 0:
            #         count_0 += 1
            #         labels = del_tensor_ele(labels, index)
            #         continue
            

            #     x_clip = x_[:pad_zeros_first_index]
            #     # x: [num_of_point, dim]
            #     if vgae.use_bn:
            #         x_clip = x_clip.unsqueeze(2)
            #         x_clip = x_clip.transpose(0, 2)             # # x: [1, dim， num_of_point]
            #         x_clip = vgae.batch_norm(x_clip)
            #         x_clip = x_clip.transpose(0, 2).squeeze()   # # x: [num_of_point, dim]
                
            #     # adj_norm, adj_label, features, norm, weight_tensor = data_transfer(adj_, x_[:pad_zeros_first_index])
            #     adj_norm, adj_label, norm, weight_tensor = data_transfer(adj_)
            #     z, A_pred = vgae(x_clip, adj_norm) 
            #     # z, A_pred = vgae(features, adj_norm) 
            #     log_lik, kl_divergence = vgae.backward(A_pred, adj_label, norm, weight_tensor)

            #     log_lik_lst.append(log_lik)
            #     kl_divergence_lst.append(kl_divergence)
            #     z_lst.append(torch.mean(z, 0, keepdim=True))
            

            ### ###### TODO:Plus classification, this u can be discounted by discount factor z
            # import pdb
            # pdb.set_trace()
            # z = torch.cat(z_lst, dim=0)
            # output = vgae.mlp_class(z)
            # _, preds = torch.max(output, 1)
            # labels_clone = labels.squeeze() - 1
            # if output.size(0) != labels_clone.size(0):
            #     labels_clone = labels_clone[:output.size(0)]

            # loss_cls = criterion(output, labels_clone)
            # writer.add_scalar('loss_cls', loss_cls.item(), (epoch+1)*i*args.batch_size_)


            # BCE = torch.tensor(log_lik_lst).mean()
            # KLD = torch.tensor(kl_divergence_lst).mean()
            # # z = torch.cat(z_lst, dim=0)
            # writer.add_scalar('count_0', count_0, (epoch+1)*i*args.batch_size_)
            # writer.add_scalar('BCE', BCE.item(), (epoch+1)*i*args.batch_size_)
            # writer.add_scalar('KLD', KLD.item(), (epoch+1)*i*args.batch_size_)
            # if args.contra_loss == '':
            #     loss_all = args.BCE_weight * BCE + args.KL_weight * KLD + loss_cls
            #     writer.add_scalar('loss_all', loss_all.item(), (epoch+1)*i*args.batch_size_)
            # else:
            #     # import pdb
            #     # pdb.set_trace()
            #     if z.shape[0] == labels.shape[0]:
            #         gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)]) # torch.Size([13])
            #         loss_mmd_lst = []
            #         loss_infonce = InfoNCE(negative_mode='unpaired')
            #         loss_infonce_lst = []
            #         query = z
            #         for j in range(3):  ## 4  tasks
            #             # if i == 45:
            #             #     import pdb
            #             #     pdb.set_trace()
            #             index_positive_keys = torch.where(labels == j)[0]
            #             index_negative_keys = torch.where(labels != j)[0]
            #             if len(index_positive_keys) > 0:
            #                 positive_keys = torch.index_select(z, 0, index_positive_keys)
            #                 negative_keys = torch.index_select(z, 0, index_negative_keys)
            #                 query = positive_keys
            #                 loss_ = loss_infonce(query, positive_keys, negative_keys)
            #                 loss_infonce_lst.append(loss_)
            #                 loss_mmd = mmd_(positive_keys, negative_keys, gammas=gammas) 
            #                 loss_mmd_lst.append(loss_mmd)
            #         loss_infonce_ = torch.stack(loss_infonce_lst).mean()
            #         loss_mmd_ = torch.stack(loss_mmd_lst).mean()
            #         writer.add_scalar('loss_infonce_', loss_infonce_.item(), (epoch+1)*i*args.batch_size_)
            #         writer.add_scalar('loss_mmd_', loss_mmd_.item(), (epoch+1)*i*args.batch_size_)
                    
            #         loss_all = args.BCE_weight * BCE + args.KL_weight * KLD + loss_cls + \
            #             args.infonce_weight * loss_infonce_ + args.disentanglement_penalty * loss_mmd_
                        


            #         vgae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            #         loss_all.backward()  # 将误差反向传播
            #         vgae_optimizer.step()  # 更新参数

            #         writer.add_scalar('loss_all', loss_all.item(), (epoch+1)*i*args.batch_size_)
            #         print('loss_all.item()', loss_all.item(), 'BCE.item()', BCE.item(), \
            #         'KLD.item()', KLD.item(), 'loss_infonce_', loss_infonce_.item(), 'loss_mmd_.item()', loss_mmd_.item())
            
            ### TODO: 1. state normalization 2. contrastive loss 3. MMD loss
        # print('the first dimension of feature shape: == 0:::', count_0)
        # path = args.save_logdir + '/' + args.log_time + '/models/encoder_epoch_' + str(epoch) + '.pt'
        # utils.create_folders_if_necessary(path)
        # torch.save(encoder, path)








