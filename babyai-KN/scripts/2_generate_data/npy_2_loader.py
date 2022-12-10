import numpy as np
import torch

import os

from torch.utils.data import TensorDataset, DataLoader

import babyai.utils as utils
from babyai.arguments import ArgumentParser
# Parse arguments
parser = ArgumentParser()
parser.add_argument("--batch_size_", type=int, default=64)
args = parser.parse_args()
args.n_actions = 7

# args.best_data_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_differ4/'
# args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/data_new/datasets_episode/'

##### 这个文件是把各个best polciy 产生的10000条episode的transitions 转换成一个npy文件，
# 最终得到的格式是：（task_num, 9999, 576, 302）,其中576是最长的episode，302是(s,a,r,s')的concat

args.best_data_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/'
args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/'
args.npy = True

obss_preprocessor = utils.ObssPreprocessor(args.best_data_logdir, None, None)
# loader_path = args.train_loader_save_logdir + 'train_loader_differ4_eps_bs' + str(args.batch_size_) + '_wo_pad'

def read_tfevent_lists(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            read_tfevent_lists(cur_path, all_files)
        else:
            all_files.append(path + "/" + file)
    return all_files

def return_npy_events_list(path):
    tfevents = read_tfevent_lists(path, [])
    npy_events_list = []
    for event in tfevents:
        if '.npy' in event:
            npy_events_list.append(event)
    return npy_events_list

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

def wo_padding_eps_data_to_max(data_x, begin, end): # 0, 63
    # length = end - begin + 1
    # pad_length = args.MAX_EPS_LEN - length
    # import pdb
    # pdb.set_trace()
    preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][0]).image) for i in range(begin, end+1)]
    data_x_1 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_2_ = torch.tensor([data_x[i][1] for i in range(begin, end+1)]) # action
    data_x_2 = list_onehot(data_x_2_, args.n_actions)
    data_x_3 = torch.tensor([data_x[i][2] for i in range(begin, end+1)]).float()  # rewards
    preprocessed_obs = [trans_image(obss_preprocessor(data_x[i][4]).image) for i in range(begin, end+1)]
    data_x_4 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_1234 = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)

    # data_all_length = data_x_1234.shape[1]
    # data_x_1234_pad = torch.nn.functional.pad(data_x_1234, (0,0,0,pad_length), mode='constant')
    return data_x_1234


def read_data_from_npy_save_loader_wo_pad(path):

    npy_lists = return_npy_events_list(path)
    index = 0
    data_X = []
    data_Y = []
    for i in npy_lists:
        index += 1
        static_below_5 = 0
        static_length_list = []
        data_x = np.load(i, allow_pickle=True) # 加载npy数据
        # data_eps_wo_pad = []
        for j in range(data_x.shape[0]):
            anchor_1, anchor_2 = 0, 0 ## begin and end position
            if data_x[j][5] == 0:
                anchor_1 = j
                for j_2 in range(j+1, data_x.shape[0]):
                    if data_x[j_2][5] == 0:
                        anchor_2 = j_2 - 1
                        break
                if anchor_2 == 0:
                    break
                j = j_2 
                wo_data_padding = wo_padding_eps_data_to_max(data_x, anchor_1, anchor_2)
                static_length_list.append(anchor_2-anchor_1)
                # print(data_padding.shape)
                data_X.append(wo_data_padding)
                data_Y.append([index])
        print('static_meng_length:', torch.tensor(static_length_list).float().mean(), i) # Open: 48.7871  Pickup 520.995  Goto  357.9526
    
    data_X_ = torch.nn.utils.rnn.pad_sequence(data_X, batch_first=True)
    train_x = data_X_.view(data_X_.shape[0], -1)
    train_y = torch.tensor(data_Y)

    if args.npy:
        # import pdb
        # pdb.set_trace()
        # data_X_ : [59994, 576, 302]
        # data_x_y = torch.stack((data_X_[0:9999], data_X_[9999:9999+9999], data_X_[9999+9999:9999+9999+9999])).numpy()
        data_x_y = torch.stack((data_X_[0:9999], data_X_[9999:9999+9999], data_X_[9999+9999:9999+9999+9999], 
                                data_X_[9999*3:9999*4], data_X_[9999*4:9999*5], data_X_[9999*5:9999*6], 
                                )).numpy()  # (6, 9999, 576, 302)
        np.save(args.train_loader_save_logdir + 'data_6_task_eps.npy', data_x_y, allow_pickle=True)
    else:
        Train_DS = TensorDataset(train_x, train_y)
        Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=args.batch_size_)
        loader_path = ''
        torch.save(Train_DL, loader_path)
        print('load and save done....')
        return loader_path

read_data_from_npy_save_loader_wo_pad(args.best_data_logdir)
