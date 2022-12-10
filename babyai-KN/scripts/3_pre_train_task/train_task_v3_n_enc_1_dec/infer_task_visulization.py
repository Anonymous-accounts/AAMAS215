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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### 区分 log
loader_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/pre_train_task/datasets/train_loader_bs128'
save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/pre_train_task/train_task_v3_n_enc_1_dec/logs_models_v3_MMD/'

path_root_ = save_logdir + '[06-10]21.48.22KL_mul_1_MMD_True'

path_root_models_ = path_root_ + '/models/vae_epoch_90.pt'
path_root_npy_ = path_root_ + '/data_z_labels.npy'
path_root_pdf_ = path_root_ + '/data_z_labels_2d.pdf'

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()
args.save_logdir = save_logdir
utils.seed(args.seed)


def infer_task_z_save():
    print('initialize the model...')
    args.latent_dim = 147
    args.n_agents = 1
    args.n_actions = 7
    args.hidden_dim = 128
    # vae = _autoencoder(args)
    # vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    vae = torch.load(path_root_models_)

    print('begin to train the task representation')

    ### read data_from_dataloader
    train_loader = torch.load(loader_path)

    data_X = []
    data_Y = []

    for i, (images, labels) in enumerate(train_loader):  # 100*32
        print("minibatch of traning vae ------", i)
        s_t_1, a, r, s_t_2 = images[:, :147], images[:, 147:147 + args.n_actions], images[:,
                                                                                   [147 + +args.n_actions]], images[:,
                                                                                                             -147:]  # s,a,r,s'
        s_t_1_, a_, r_, s_t_2_, mu, var, z = vae(s_t_1, a, r, s_t_2, labels)
        data_X.append(z)
        data_Y.append(labels)

    data_X_ = torch.cat(data_X, dim=0)
    data_Y_ = torch.cat(data_Y, dim=0)[:, None].float()
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
    index = 0
    for i in mask:
        arr = torch.index_select(torch.from_numpy(data), 0, torch.from_numpy(np.where(data[:, -1] == i)[0]))
        arr_index = torch.randint(0, arr.shape[0], (number,))
        arr = arr[arr_index].numpy()
        index += int(tmp[i])
        list_.append(arr.tolist())
    print(np.array(list_).shape)
    return np.array(list_).reshape(-1, np.array(list_).shape[-1])


def plot_clustering():
    # path_root_ = './logs_models/[06-05]21.52.20/'
    # path_root_ = './logs_models/KL_mul_0.1_[06-05]23.22.08/'
    # path_root_ = './logs_models/KL_mul_0.1_contra_loss_[06-06]00.12.17/'
    # path_root_npy_ = path_root_ + '/data_z_labels.npy' #
    # path_root_npy_ = path_root_ + 'data_z_labels_KL_mul_0.1.npy'
    # path_root_npy_ = path_root_ + 'data_z_labels_KL_mul_0.1_contra_loss.npy'

    data_all = np.load(path_root_npy_)
    data_tmp = data_pick(data_all, 200)
    iris = chj_load_file(data_tmp)
    # iris = chj_load_file(data_all)
    X_tsne = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(iris.data)
    fig = plt.figure()
    plt.axis('off')  # Remove the axis
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target, edgecolors='k', marker='o', linewidths=1.0, cmap='Spectral')

 
    # for i in range(X_tsne.shape[0]): # Draw labels for each data point in the plot
    #     plt.text(X_tsne[i, 0], X_tsne[i, 1], str(iris.target[i]), color=plt.cm.Set1(iris.target[i] / 10), fontdict={'weight': 'bold', 'size': 7})


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
    # infer_task_z_save()
    # plot the results
    plot_clustering()
    # plot_clustering_3d()



















