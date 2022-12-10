import torch
model_ = torch.load('/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v6_corro/logs_models_v6/[06-30]15.59.04train_loader_bs_64_corro_others_neg_/models/encoder_epoch_1150000.pt')
model_2 = torch.load('/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v6_corro/logs_models_v6_4_tasks_new/[07-19]10.49.32train_loader_bs_64_corro_others_neg_/models/encoder_epoch_1180000.pt')
import pdb
pdb.set_trace()
print(model_)

