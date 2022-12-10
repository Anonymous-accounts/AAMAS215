from abc import ABC, abstractmethod
import torch
import numpy

from babyai.models.tree import TaskTree
from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from torch.distributions import Categorical

from torchkit import pytorch_utils as ptu

class BaseAlgo_Ours_v2(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, encoder_tuple, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel

        self.query_encoder = encoder_tuple[0]
        self.pre_subtask_encoder = encoder_tuple[1]
        self.att = encoder_tuple[2]

        self.acmodel.train()
        self.query_encoder.train()
        self.pre_subtask_encoder.train()
        self.att.train()
        
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.n_actions = 7
        self.N = 1
    
        self.task_tree = TaskTree(self.num_procs, self.N)

        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.task_embss = torch.zeros(*shape, 5, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        
        self.nodes = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.edges = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        
        self.exps = DictList()

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        
        reconstrction_flag = True
        last_act = torch.zeros(self.num_procs, self.n_actions).cuda()
        last_r = torch.zeros(self.num_procs, 1).cuda()
        for i in range(self.num_frames_per_proc): # 40
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device) # 64*dict --> {'image': 64,7,7,3 'instr': 64,5}
            
            if self.task_tree.tree_visited[0][-1]: #### 都访问完了
                reconstrction_flag = True
            if reconstrction_flag:
                self.task_tree.init()
                self.task_tree.construct_tree(self.query_encoder, self.pre_subtask_encoder, self.att, self.acmodel, preprocessed_obs, self.memory)
                reconstrction_flag = False
                ######### Top-k N-step tree construction  ############ k=1, N=3 #############
                # task_tree.tree_node, task_tree.tree_edges, task_tree.tree_visited
                self.task_tree.tree_visited[0][-1] = 1 #### N=1 always reconstruct
            
            self.task_tree.last_act, self.task_tree.last_r = last_act, last_r

            with torch.no_grad():

                # model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), task_tree) # memeory 64*256  mask 64*1  --> {'dist':Categorical [64,7]; 'value' 64,7 'memory': 64,256 'ex..'}
                model_results = self.acmodel.forward_v2(preprocessed_obs, self.memory * self.mask.unsqueeze(1), self.task_tree)

                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']
                task_embs = model_results['task_embs']

            action = dist.sample() # 64,

            obs, reward, done, env_info = self.env.step(action.cpu().numpy()) # action: ndarray(64,)  reward&done: tuple(64,)

            last_act = torch.zeros(self.num_procs, self.n_actions).cuda().scatter_(1, action[:,None], 1) # 2.7
            last_r = torch.tensor(reward, device=self.device)[:,None]

            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values      task_embs

            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.task_embss[i] = task_embs
            self.nodes[i] = self.task_tree.tree_node[:, self.task_tree.achor].cuda()
            self.edges[i] = self.task_tree.tree_edges[:, self.task_tree.achor].cuda()

            self.task_tree.achor += 1
            

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        # del task_tree
        
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), self.task_tree)['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        # exps = DictList()
        self.exps.clear()
        self.exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        self.exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        self.exps.task_embs = self.task_embss.transpose(0, 1).reshape(-1, *self.task_embss.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        self.exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        self.exps.action = self.actions.transpose(0, 1).reshape(-1)
        self.exps.value = self.values.transpose(0, 1).reshape(-1)
        self.exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        self.exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        self.exps.returnn = self.exps.value + self.exps.advantage
        self.exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        self.exps.nodes = self.nodes.transpose(0, 1).reshape(-1)
        self.exps.edges = self.edges.transpose(0, 1).reshape(-1)

        if self.aux_info:
            self.exps = self.aux_info_collector.end_collection(self.exps)

        # Preprocess experiences

        self.exps.obs = self.preprocess_obss(self.exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return self.exps, log

    @abstractmethod
    def update_parameters(self):
        pass
