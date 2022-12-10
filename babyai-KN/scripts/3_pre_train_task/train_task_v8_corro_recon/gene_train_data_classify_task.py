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

# args.save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/train_task_v4_vgae/logs_models_v4_differ4/'
# args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/datasets_episode/'
# args.best_data_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick/best_data_differ4'
args.best_data_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/'
args.train_loader_save_logdir = '/data2/username_high/username/BABYAI/data_new/best_data_new/datasets_npy/'

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


def wo_padding_transition_data_to_max(data_x):
    preprocessed_obs = [trans_image(obss_preprocessor(data_x[0]).image)]
    data_x_1 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_2_ = torch.tensor([data_x[1]]) # action
    data_x_2 = list_onehot(data_x_2_, args.n_actions)
    data_x_3 = torch.tensor([data_x[2]]).float()  # rewards
    preprocessed_obs = [trans_image(obss_preprocessor(data_x[4]).image)]
    data_x_4 = torch.cat(preprocessed_obs).reshape(len(preprocessed_obs), -1)
    data_x_1234 = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)
    return data_x_1234.numpy().tolist()[0]  #### (1, 302)


def read_data_from_npy_save_loader_wo_pad(path):

    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-Open-v0_best_data/data_10000.npy
    # (497826, 6)
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-Pickup-v0_best_data/data_10000.npy
    # (782069, 6)
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-GoTo-v0_best_data/data_10000.npy
    # (727298, 6)
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-UnblockPickup-v0_best_data/data_10000.npy
    # (1863820, 6)
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-PickupLoc-v0_best_data/data_10000.npy
    # (79660, 6)
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_differ4/BabyAI-PutNextLocal-v0_best_data/data_10000.npy
    # (126887, 6)

    npy_lists = return_npy_events_list(path)
    index = 0
    data_X = [] # Open Pickup Goto # (497826, 6) # (5220010, 6) # (3589743, 6)
    data_Y = []
    max_transitions_num = 400000
    for i in npy_lists:
        index += 1
        print(i)
        print(index)
        npy_lists_make_data_x, npy_lists_make_data_y = [], []
        data_x = np.load(i, allow_pickle=True) # 加载npy数据
        # print(data_x.shape)
        for j in range(max_transitions_num):
            wo_data_padding = wo_padding_transition_data_to_max(data_x[j])
            npy_lists_make_data_x.append(wo_data_padding)
            npy_lists_make_data_y.append([index])
        data_X.append(npy_lists_make_data_x)
        data_Y.append(npy_lists_make_data_y)
        
    data_X_ = np.array(data_X)   # (3, 400000, 302)
    data_Y_ = np.array(data_Y)   # (3, 400000, 1)

    # import pdb
    # pdb.set_trace()
    data_x_y = np.concatenate((data_X_, data_Y_), axis=-1)  # (3, 400000, 303)
    print(data_x_y.shape)
    
    np.save(args.train_loader_save_logdir + '/data_4_task_new_Open_Goto20000_UnblockPickup_PutNextLocal.npy', data_x_y, allow_pickle=True)

    
    ### data_4_task_new_Open_Goto_UnblockPickup_PutNextLocal.npy
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-Open-v0_best_data/data_20000.npy
    # 1
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-GoTo-v0_best_data/data_10000.npy
    # 2
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-UnblockPickup-v0_best_data/data_10000.npy
    # 3
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-PutNextLocal-v0_best_data_50000/data_50000.npy
    # 4
        
    ### data_3_task_new_Open_UnblockPickup_PutNextLocal.npy
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-Open-v0_best_data/data_20000.npy
    # 1
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-UnblockPickup-v0_best_data/data_10000.npy
    # 2
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick_20220720/BabyAI-PutNextLocal-v0_best_data/data_10000.npy
    # 3

    ##### data_4_task_new.npy
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick/best_data_differ4/BabyAI-Open-v0_best_data/data_10000.npy
    # 1
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick/best_data_differ4/BabyAI-Pickup-v0_best_data/data_10000.npy
    # 2
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick/best_data_differ4/BabyAI-GoTo-v0_best_data/data_10000.npy
    # 3
    # /data2/username_high/username/BABYAI/data_new/best_data_new/best_data_pick/best_data_differ4/BabyAI-PutNextLocal-v0_best_data_50000
    # 4
    # 

    # train_x = data_X.view(data_X.shape[0], -1)
    # train_y = torch.tensor(data_Y)
    # Train_DS = TensorDataset(train_x, train_y)
    # Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=args.batch_size_)
    # import pdb
    # pdb.set_trace()
    # torch.save(Train_DL, loader_path)
    print('load and save done....')
    return loader_path

best_data_path = args.best_data_logdir
loader_path = read_data_from_npy_save_loader_wo_pad(best_data_path)
