import torch
import torch.nn as nn
from torch.distributions import Categorical
import babyai.rl
import math
import numpy as np

from treelib import Tree, Node

class TaskTree(nn.Module):
    def __init__(self, num_procs, N, K, args):
        super(TaskTree, self).__init__()
        self.num_procs = num_procs
        self.N = N ## 深度  nodes == (k^n-1)/k-1
        self.K = K ## 宽度
        # self.node_nums = (K**N-1)//(K-1)
        self.n_actions = 7
        self.tree_node = torch.zeros(self.num_procs, self.N).cuda()
        self.tree_edges = torch.zeros(self.num_procs, self.N).cuda()
        self.tree_visited = torch.zeros(self.num_procs, self.N).cuda()
        # self.sample_dist = torch.zeros(self.num_procs, 0).cuda()
        
        self.all_tree_node = torch.zeros(self.num_procs, self.K**self.N-1).cuda()
        self.all_tree_edges = torch.zeros(self.num_procs, self.K**self.N-1).cuda()   
        
        self.last_act, self.last_r = None, None
        # self.achor = 0
        self.achor_node = torch.zeros(self.num_procs).cuda()
        self.achor_edge = torch.zeros(self.num_procs).cuda()
        self.subtask_duration_rewards = torch.zeros(self.num_procs).cuda()
        
        self.args = args

    def init(self):
        self.tree_node.zero_()
        self.tree_edges.zero_()
        self.tree_visited.zero_()
        self.tree_visited[:, 0] = 1
        self.achor_node.zero_()
        self.achor_edge.zero_()
        self.subtask_duration_rewards.zero_()
        # self.sample_dist.zero_()
        # self.achor = 0
        # self.tree = Tree()
    
    def set_tree(self, query_encoder, pre_subtask_encoder, att):
        self.query_encoder, self.pre_subtask_encoder, self.att = query_encoder, pre_subtask_encoder, att

    
    def sample_top_k(self, query_probs, K):
        log_pi, subtasks, sample_subtask = [], [], None
        query_probs = query_probs + 1e-7
        sample_dist = Categorical(query_probs)
        for i in range(K):
            if sample_subtask is not None:
                inf_pad = torch.tensor(torch.zeros_like(sample_subtask[:,None]), dtype=torch.float32)
                query_probs.scatter_(1, sample_subtask[:,None], inf_pad)
                # print(torch.min(query_probs))
                sample_dist_new = Categorical(query_probs)
                sample_subtask = sample_dist_new.sample()
                log_pi.append(sample_dist.log_prob(sample_subtask))
                subtasks.append(sample_subtask) 
            else:
                sample_subtask = sample_dist.sample()
                log_pi.append(sample_dist.log_prob(sample_subtask))
                subtasks.append(sample_subtask)
        return torch.stack(log_pi).permute(1,0), torch.stack(subtasks).permute(1,0) # torch.Size([64, 3]), torch.Size([64, 3])
                

    
    ### compute ucb-score
    def compute_ucb(self, count_i, count_par, mean_r):
        return mean_r + 0.5 * math.sqrt(2*math.log(count_par) / count_i)
    
    def construct_tree(self, query_encoder, pre_subtask_encoder, att, acmodel, preprocessed_obs, memory):
        self.query_encoder, self.pre_subtask_encoder, self.att, self.acmodel = query_encoder, pre_subtask_encoder, att, acmodel
        preprocessed_obs_instr, cur_memory = preprocessed_obs.instr, memory
        flatten_obs = torch.transpose(torch.transpose(preprocessed_obs.image, 1, 3), 2, 3).reshape(self.num_procs, -1) # 64.147
        last_act = torch.zeros(self.num_procs, self.n_actions).cuda()
        last_r = torch.zeros(self.num_procs, 1).cuda()
        for i in range(self.N):
            with torch.no_grad():   
                if i == 0:
                    query_emb = query_encoder(flatten_obs, last_act, last_r)   # torch.Size([1, 5])
                    pre_subtask_embs = pre_subtask_encoder(flatten_obs, last_act, last_r) # torch.Size([1, 4, 5])
                    query_probs = att.att_prob(query_emb[:,None], pre_subtask_embs) # 2.4.1
                    query_probs = query_probs + 10e-7
                    sample_dist = Categorical(query_probs[:,:,0])
                    sample_subtask = sample_dist.sample() # ## torch.Size([2])    tensor([2, 0], device='cuda:0')
                    sample_probs = torch.gather(query_probs[:,:,0], 1, sample_subtask[:,None]) # 2.1
                    #### add node , add edge
                    self.all_tree_node[:, i] = sample_subtask
                    # self.tree_edges[:, i] = sample_probs[:,0]
                    self.all_tree_edges[:, i] = sample_dist.log_prob(sample_subtask)
                    
                    for ind_ in sample_subtask: ### add count
                        self.args.subtask_count[ind_] += 1
                    
                    ##### concat the task embeddings
                    query_task_emb = torch.gather(pre_subtask_embs, -1, sample_subtask[:,None,None].repeat(1,1,5))[:,0]  # 64, 5
                    query_all = torch.cat((query_emb, query_task_emb.detach()), dim=-1)
                    pred_obs = query_encoder.pred(query_all)  # 2.147  # sample_subtask[:,None,None]
                    pred_obs_res = pred_obs.reshape(*(preprocessed_obs.image.shape)) # 2.7.7.3

                    cur_preprocessed_obs = babyai.rl.DictList()
                    cur_preprocessed_obs.image, cur_preprocessed_obs.instr = pred_obs_res, preprocessed_obs_instr

                    pred_model_results = acmodel(cur_preprocessed_obs, cur_memory)  ### TODO: self.task_tree
                    pred_dist = pred_model_results['dist']
                    cur_memory = pred_model_results['memory']
                    pred_act = pred_dist.sample() # torch.Size([2])
                    pred_act_action = torch.zeros(self.num_procs, self.n_actions).cuda().scatter_(1, pred_act[:,None], 1) # 2.7
                    flatten_obs, last_act = pred_obs, pred_act_action  ### construct the node one by one
                    
                elif i == 1:
                    query_emb = query_encoder(flatten_obs, last_act, last_r)   # torch.Size([1, 5])
                    pre_subtask_embs = pre_subtask_encoder(flatten_obs, last_act, last_r) # torch.Size([1, 4, 5])
                    query_probs = att.att_prob(query_emb[:,None], pre_subtask_embs) # 2.4.1
                    # print(query_probs.shape)
                    values, indices = self.sample_top_k(query_probs[:,:,0], self.K)
                    # values, indices = torch.topk(query_probs, self.K, dim=1)
                        
                    #### add node , add edge
                    self.all_tree_node[:, 1:1+self.K] = indices
                    # self.tree_edges[:, i] = sample_probs[:,0]
                    self.all_tree_edges[:, 1:1+self.K] = values
                    
                    
                    for ind_ in indices.flatten(): ### add count
                        self.args.subtask_count[ind_] += 1
                    
                    index_now = 1 + self.K
                else:
                    #### find last task,,embs
                    for j in range(self.K**(i-1)):  ### 每层开始的index: (K**N-1)//(K-1)
                        index_cur = self.all_tree_node[:, (self.K**(i-1)-1)//(self.K-1)+j] 
                        query_task_emb =  torch.gather(pre_subtask_embs, -1, torch.tensor(index_cur[:,None,None].repeat(1,1,5), dtype=torch.int64))[:,0]
                        query_all = torch.cat((query_emb, query_task_emb.detach()), dim=-1)
                        pred_obs = query_encoder.pred(query_all)  # 2.147
                        pred_obs_res = pred_obs.reshape(*(preprocessed_obs.image.shape)) # 2.7.7.3
                    
                        cur_preprocessed_obs = babyai.rl.DictList()
                        cur_preprocessed_obs.image, cur_preprocessed_obs.instr = pred_obs_res, preprocessed_obs_instr

                        pred_model_results = acmodel(cur_preprocessed_obs, cur_memory)  ### TODO: self.task_tree
                        pred_dist = pred_model_results['dist']
                        cur_memory = pred_model_results['memory']
                        pred_act = pred_dist.sample() # torch.Size([2])
                        pred_act_action = torch.zeros(self.num_procs, self.n_actions).cuda().scatter_(1, pred_act[:,None], 1) # 2.7
                        flatten_obs, last_act = pred_obs, pred_act_action  ### construct the node one by one
                        
                        query_emb = query_encoder(flatten_obs, last_act, last_r)   # torch.Size([1, 5])
                        pre_subtask_embs = pre_subtask_encoder(flatten_obs, last_act, last_r) # torch.Size([1, 4, 5])
                        query_probs = att.att_prob(query_emb[:,None], pre_subtask_embs) # 2.4.1
                        # print(query_probs.shape)
                        # values, indices = torch.topk(query_probs, self.K, dim=1)
                        values, indices = self.sample_top_k(query_probs[:,:,0], self.K)
                        #### add node , add edge
                        self.all_tree_node[:, index_now:index_now+self.K] = indices
                        # self.tree_edges[:, i] = sample_probs[:,0]
                        self.all_tree_edges[:, index_now:index_now+self.K] = values
                    
                    
                        for ind_ in indices.flatten(): ### add count
                            self.args.subtask_count[ind_] += 1
                            
                        index_now = 1 + self.K
                
        # print(self.all_tree_node)             
        # print(self.all_tree_edges)
        
        ##### 剪枝 random walk 
        if self.args.random_walk:
            for i in range(self.N):
                if i == 0:
                    self.tree_node[:,i] = self.all_tree_node[:,i]
                    self.tree_edges[:,i] = self.all_tree_edges[:,i]
                else:    
                    layer_begin = (self.K**(i-1)-1)//(self.K-1)
                    layer_num = self.K**i
                    rand_index = torch.randint(layer_begin, layer_begin+layer_num, (1,))[0]
                    self.tree_node[:,i] = self.all_tree_node[:,rand_index]
                    self.tree_edges[:,i] = self.all_tree_edges[:,rand_index]
                
        ###### discounted-max-path-length
        elif self.args.dis_max_path:
            if self.N == 1:
                self.tree_node[:,0] = self.all_tree_node[:,0]
                self.tree_edges[:,0] = self.all_tree_edges[:,0]
            else:
                gamma_ = 0.8
                ### generate path first
                tree = Tree()
                index = 0 
                tree.create_node(str(index), str(index)) # root node
                for i in range(1, self.N, 1):#### parent ：：：(index-1)//K # the index for cur layer
                    next_i = i+1
                    layer_begin = (self.K**(next_i-1)-1)//(self.K-1)
                    layer_end = layer_begin + self.K**i   # [layer_begin, layer_end)
                    for j in range(layer_begin, layer_end, 1):
                        tree.create_node(str(j), str(j), parent=str((j-1)//self.K))
                # tree.show()
                # print(tree.paths_to_leaves())
                paths = [list(map(int, i)) for i in tree.paths_to_leaves()]
                ucbs = np.zeros((len(paths), 1))
                
                for sub_pro in range(self.args.procs): 
                    dis_max_path_score, dis_max_path_index = 0, 0
                    for i in range(len(paths)): #
                        for j in range(1, len(paths[i]), 1):
                            cur_node, par_node = self.all_tree_node[sub_pro,paths[i][j]].int().item(), self.all_tree_node[sub_pro, paths[i][j-1]].int().item()
                            ucbs[i][0] += gamma_**j * self.all_tree_edges[sub_pro, par_node] * self.compute_ucb(self.args.subtask_count[cur_node], self.args.subtask_count[par_node], self.args.subtask_mean_rew[cur_node])
                        if ucbs[i][0] > dis_max_path_score:
                            dis_max_path_score = ucbs[i][0] 
                            dis_max_path_index = i  
                    
                    for i_ in range(self.args.K):     
                        self.tree_node[sub_pro, i_] = self.all_tree_node[sub_pro, paths[dis_max_path_index][i_]]
                        self.tree_edges[sub_pro, i_] = self.all_tree_edges[sub_pro, paths[dis_max_path_index][i_]]
                        
            

    # def construct_tree(self, query_encoder, pre_subtask_encoder, att, acmodel, preprocessed_obs, memory):
    #     self.query_encoder, self.pre_subtask_encoder, self.att, self.acmodel = query_encoder, pre_subtask_encoder, att, acmodel
    #     preprocessed_obs_instr, cur_memory = preprocessed_obs.instr, memory
    #     flatten_obs = torch.transpose(torch.transpose(preprocessed_obs.image, 1, 3), 2, 3).reshape(self.num_procs, -1) # 64.147
    #     last_act = torch.zeros(self.num_procs, self.n_actions).cuda()
    #     last_r = torch.zeros(self.num_procs, 1).cuda()
    #     for i in range(self.N):
    #         with torch.no_grad():
    #             query_emb = query_encoder(flatten_obs, last_act, last_r)   # torch.Size([1, 5])
    #             pre_subtask_embs = pre_subtask_encoder(flatten_obs, last_act, last_r) # torch.Size([1, 4, 5])
    #             query_probs = att.att_prob(query_emb[:,None], pre_subtask_embs) # 2.4.1
    #             query_probs = query_probs + 10e-7
    #             sample_dist = Categorical(query_probs[:,:,0])
    #             sample_subtask = sample_dist.sample() # ## torch.Size([2])    tensor([2, 0], device='cuda:0')
    #             sample_probs = torch.gather(query_probs[:,:,0], 1, sample_subtask[:,None]) # 2.1

    #             #### add node , add edge
    #             self.tree_node[:, i] = sample_subtask
    #             # self.tree_edges[:, i] = sample_probs[:,0]
    #             self.tree_edges[:, i] = sample_dist.log_prob(sample_subtask)

    #             pred_obs = query_encoder.pred(query_emb)  # 2.147
    #             pred_obs_res = pred_obs.reshape(*(preprocessed_obs.image.shape)) # 2.7.7.3

    #             cur_preprocessed_obs = babyai.rl.DictList()
    #             cur_preprocessed_obs.image, cur_preprocessed_obs.instr = pred_obs_res, preprocessed_obs_instr

    #             pred_model_results = acmodel(cur_preprocessed_obs, cur_memory)  ### TODO: self.task_tree
    #             pred_dist = pred_model_results['dist']
    #             cur_memory = pred_model_results['memory']
    #             pred_act = pred_dist.sample() # torch.Size([2])
    #             pred_act_action = torch.zeros(self.num_procs, self.n_actions).cuda().scatter_(1, pred_act[:,None], 1) # 2.7
    #             flatten_obs, last_act = pred_obs, pred_act_action  ### construct the node one by one
