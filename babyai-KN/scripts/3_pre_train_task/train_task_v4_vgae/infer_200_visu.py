#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import numpy as np
import torch
# from info_nce import InfoNCE

import babyai
import babyai.utils as utils
from babyai.arguments import ArgumentParser
# from babyai.models.vae import _autoencoder
# from utils.recursive_read import return_npy_events_list
# from torch.utils.data import TensorDataset, DataLoader
from babyai.models.vgae import VGAE, data_transfer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### 区分 log
loader_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/datasets_episode/train_loader_differ4_eps_bs64_wo_pad'
save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/train_task_v4_vgae/logs_models_v4_differ4/'

path_root_ = save_logdir + '[06-15]11.23.02_bs_64_wo_padBCE_5.0KL_0.011000.0MMD_10.0'
# [06-13]10.51.30_bs_64_wo_padBCE_5.0KL_0.01contra_loss_1000.0MMD_10.0

path_root_models_ = path_root_ + '/models/vae_epoch_49.pt'
path_root_npy_ = path_root_ + '/data_z_labels_mean_norm.npy'
path_root_pdf_ = path_root_ + '/data_z_labels_2d_mean_norm.pdf'

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()
args.save_logdir = save_logdir
args.batch_size_ = 64 ### !!!!!!!! modify here
args.actual_max_eps_len = 576 ### !!!!!!!! modify here
args.feature_length = 302
args.MAX_EPS_LEN = 16 ### !!!!!!!! modify here
utils.seed(args.seed)



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

def infer_task_z_save():
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 128
    # vae = _autoencoder(args)
    # vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    vgae = torch.load(path_root_models_)

    print('begin to train the task representation')

    ### read data_from_dataloader
    train_loader = torch.load(loader_path)

    data_X = []
    data_Y = []

    for i, (images, labels) in enumerate(train_loader):  # 100*32
        # import pdb
        # pdb.set_trace()
        if images.view(-1).shape[0] != args.batch_size_ * args.actual_max_eps_len * args.feature_length:
                continue
        print("minibatch of traning vae ------", i)

        train_x = images.view(args.batch_size_, args.actual_max_eps_len, -1) # 64 46 302
        for index in range(train_x.shape[0]):
            # print(index)
            # if i == 8 and index == 61 :
            #     import pdb
            #     pdb.set_trace()
            x_ = train_x[index] ### 64, 302 可以normalize
            adj_, pad_zeros_first_index = generate_seq_adj(x_, args.actual_max_eps_len)
            if x_[:pad_zeros_first_index].shape[0] == 0:
                    # count_0 += 1
                    # labels = del_tensor_ele(labels, index)
                    continue
            adj_norm, adj_label, norm, weight_tensor = data_transfer(adj_)
            features = x_[:pad_zeros_first_index]
            z, A_pred = vgae(features, adj_norm) 
            
            # data_X.append(z.view(1, -1))
            data_X.append(torch.mean(z, 0, keepdim=True)) # 
            # data_X.append(z[[-1], :]) # 
            data_Y.append(labels[[index]])

    
    # import pdb
    # pdb.set_trace()
    data_X_ = torch.cat(data_X, dim=0)
    data_Y_ = torch.cat(data_Y, dim=0).float()
    data_all = torch.cat((data_X_, data_Y_), dim=1).detach().numpy()

    np.save(path_root_npy_, data_all)


class chj_data(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

def chj_load_file(fdata):
    feature = fdata[:, :-1]
    print(feature.shape)
    target = fdata[:, -1]
    print(target.shape)
    res = chj_data(feature, target)
    return res


def data_pick(data, number):
    # import pdb
    # pdb.set_trace()
    # data_data = data_normal_2d(torch.from_numpy(data[:, :-1])).numpy() ### normalization 一下
    # data = np.concatenate((data_data, data[:,[-1]]), axis=1)
    mask = np.unique(data[:,-1])
    # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24.]
    tmp = {}
    for v in mask:
        tmp[v] = np.sum(data[:, -1] == v)
    # print("statistical results：")
    # print(tmp)
    ###
    # {0: 163506, 1: 29840, 2: 18064,   3: 76560,  4: 35257, 5: 62005,
    #  6: 54210,  7: 57096, 8: 124754,  9: 48969,  10: 110304,
    # 11: 118840, 12: 34929, 13: 20456, 14: 19640, 15: 53170,
    # 16: 24516,  17: 36394, 18: 26945, 19: 79088, 20: 9000,
    # 21: 32273,  22: 25562, 23: 23939, 24: 30720}
    list_ = []
    pick_random = False
    for i in mask:
        if pick_random: # 随机挑
            arr = torch.index_select(torch.from_numpy(data), 0, torch.from_numpy(np.where(data[:, -1] == i)[0]))
            arr_index = torch.randint(0, arr.shape[0], (number,))
            arr = arr[arr_index].numpy()
            # index += int(tmp[i])
        else: # 
            arr = torch.index_select(torch.from_numpy(data), 0, torch.from_numpy(np.where(data[:, -1] == i)[0]))
            arr = arr[:number].numpy()

        list_.append(arr.tolist())
    print(np.array(list_).shape)

    data_tmp = np.array(list_).reshape(-1, np.array(list_).shape[-1])

    data_data = data_normal_2d(torch.from_numpy(data_tmp[:, :-1])).numpy() ### normalization 一下
    data_tmp = np.concatenate((data_data, data_tmp[:,[-1]]), axis=1)
    # delete 259, 273, 280, 359, 369
    
    # data_tmp = np.delete(data_tmp, 259, 0) 
    # data_tmp = np.delete(data_tmp, 272, 0) 
    # data_tmp = np.delete(data_tmp, 278, 0) 
    # data_tmp = np.delete(data_tmp, 356, 0) 
    # data_tmp = np.delete(data_tmp, 365, 0) 

    # data_tmp = np.delete(data_tmp, 548, 0) 
    # data_tmp = np.delete(data_tmp, 565, 0) 

    return data_tmp


def data_pooling(data_tmp):
    # import pdb
    # pdb.set_trace()
    length = data_tmp.shape[0]
    feature = data_tmp[:, :-1]
    feature = np.array([np.average(i.reshape(args.MAX_EPS_LEN, -1), axis=1) for i in feature])
    target = data_tmp[:, [-1]]
    data_tmp = np.concatenate((feature, target), axis=1)

    return data_tmp

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


def plot_clustering():
    # path_root_ = './logs_models/[06-05]21.52.20/'
    # path_root_ = './logs_models/KL_mul_0.1_[06-05]23.22.08/'
    # path_root_ = './logs_models/KL_mul_0.1_contra_loss_[06-06]00.12.17/'
    # path_root_npy_ = path_root_ + '/data_z_labels.npy' #
    # path_root_npy_ = path_root_ + 'data_z_labels_KL_mul_0.1.npy'
    # path_root_npy_ = path_root_ + 'data_z_labels_KL_mul_0.1_contra_loss.npy'
    # import pdb
    # pdb.set_trace()
    data_all = np.load(path_root_npy_)
    data_tmp = data_pick(data_all, 200)
    # data_tmp = data_pooling(data_tmp)
    iris = chj_load_file(data_tmp)
    # iris = chj_load_file(data_all)
    X_tsne = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(iris.data)
    fig = plt.figure()
    plt.axis('off')  # Remove the axis
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target, edgecolors='k', marker='o', linewidths=1.0, cmap='Spectral')

 
    # for i in range(X_tsne.shape[0]): # Draw labels for each data point in the plot
    #     plt.text(X_tsne[i, 0], X_tsne[i, 1], str(iris.target[i]), color=plt.cm.Set1(iris.target[i] / 10), fontdict={'weight': 'bold', 'size': 7})
    

    # for i in range(X_tsne.shape[0]): # Draw labels for each data point in the plot
    #     plt.text(X_tsne[i, 0], X_tsne[i, 1], str(i), color=plt.cm.Set1(i / 10), fontdict={'weight': 'bold', 'size': 7})


    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0.03, 0.03)
    # plt.axis('off')
    # plt.show()
    # fig.savefig(path_root_ + '/data_z_labels_naive_plot.pdf', dpi=600, format='pdf')
    # fig.savefig(path_root_ + './data_z_labels_KL_mul_0_1_plot.pdf', dpi=600, format='pdf')
    # fig.savefig(path_root_ + './data_z_labels_KL_mul_0_1_contra_loss_plot.pdf', dpi=600, format='pdf')
    fig.savefig(path_root_pdf_, dpi=600, format='pdf')


def plot_clustering_3d():

    data_all = np.load(path_root_npy_)
    data_tmp = data_pick(data_all, 200)
    iris = chj_load_file(data_tmp)
    # iris = chj_load_file(data_all)

    X_tsne = TSNE(n_components=3).fit_transform(iris.data)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=iris.target, linewidths=0.5, edgecolors='k', cmap='Spectral')
    # target = np.arange(0, 78, 1)
    # target_txt = [str(i) for i in target]
    # for i in range(len(target_txt)):
    #     ax.text(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], target_txt[i],
    #             color='b', fontsize='10', weight='bold')

    
    # plt.axis('off')  # Remove the axis
    plt.show()
    fig.savefig(path_root_pdf_, dpi=600, format='pdf')



if __name__ == '__main__':
    infer_task_z_save()
    # plot the results
    plot_clustering()
    # plot_clustering_3d()



















