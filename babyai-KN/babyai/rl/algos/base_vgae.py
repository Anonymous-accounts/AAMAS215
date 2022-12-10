from abc import ABC, abstractmethod
import torch
import numpy as np

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
import datetime

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
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
        self.acmodel.train()
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
        # self.device = torch.device("cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        

        self.vgae_optimizer = torch.optim.Adam(self.acmodel.vgae.parameters(), self.lr, (0.9, 0.999), eps=0.95)


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()

        self.preprocessed_obs_ = None
        self.action_ = torch.zeros([self.num_procs]).to(self.device)
        self.reward_ = torch.zeros([self.num_procs]).to(self.device)
        self.preprocessed_obs_next_ = self.preprocessed_obs_

        self.obss = [None]*(shape[0])
        self.z_tensor = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        
        self.z_tensors = torch.zeros(*shape, self.acmodel.vgae_hidden2_dim, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.log_lik = torch.zeros(*shape, device=self.device)
        self.kl_diverge = torch.zeros(*shape, device=self.device)

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

        self.vage_log_lik_coef = 1.
        self.vgae_kl_diverge_coef = 0.1



    def trans_image(self, image):
        return torch.transpose(torch.transpose(image, 1, 3), 2, 3)

    def list_onehot(self, actions: list, n: int) -> torch.Tensor:
        """
        列表动作值转 onehot
        actions: 动作列表
        n: 动作总个数
        """
        result = []
        for action in actions:
            result.append([int(k == action) for k in range(n)])
        result = torch.tensor(result, dtype=torch.float).to(self.device)
        return result

    def vage_processing_inputs(self, data_x): # 0, 63
        """
        concatenate 转换成 vgae的输入
        data_x: tuple(s,a,r,s')
        """
        data_x_1 = self.trans_image(data_x[0].image).reshape(self.num_procs, -1)
        data_x_2 = self.list_onehot(data_x[1], self.args.n_actions) # action
        data_x_3 = data_x[2][:, None].float().to(self.device)  # rewards
        data_x_4 = self.trans_image(data_x[3].image).reshape(self.num_procs, -1)
        data_x_1234 = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)
        return data_x_1234

    def generate_seq_adj(self, EPS_TIMESTEP):
        adj_zeros_numpy = np.zeros((EPS_TIMESTEP, EPS_TIMESTEP))
        if EPS_TIMESTEP == 1:
            adj_zeros_numpy[0][0] = 1
        for i in range(0, EPS_TIMESTEP-1):
            adj_zeros_numpy[i][i+1] = 1
        return adj_zeros_numpy
        # return torch.tensor(adj_zeros_numpy).to(self.device)

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
        vgae_inputs_list = []
        z_total_tensor_list = []
        for i in range(self.num_frames_per_proc): # 40
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device) # 64*dict --> {'image': 64,7,7,3 'instr': 64,5}
            self.preprocessed_obs_ = preprocessed_obs
            self.preprocessed_obs_next_ = self.preprocessed_obs_
            
            if self.args.use_vgae:
                # print('use_vage')
                vgae_inputs = self.vage_processing_inputs((self.preprocessed_obs_, self.action_, self.reward_, self.preprocessed_obs_next_)) # torch.Size([64, 302])
                vgae_inputs_list.append(vgae_inputs)
                vgae_inputs_tensor = torch.stack(vgae_inputs_list, dim=1) # torch.Size([64, 1, 302])
                EPS_TIMESTEP = len(vgae_inputs_list)
                adj_ = self.generate_seq_adj(EPS_TIMESTEP)
                
                import datetime
                # starttime = datetime.datetime.now()

                adj_norm, adj_label, norm, weight_tensor = self.acmodel.vgae.data_transfer(adj_)
                adj_lists = [self.acmodel.vgae.transfer_edge_index(adj_norm.to_dense()) for _ in range(self.num_procs)]
                # with torch.autograd.detect_anomaly():
                z, A_pred = self.acmodel.vgae.forward_batch(vgae_inputs_tensor, adj_lists)

                log_lik_lists, kl_divergence_lists = 0, 0
                for j in range(self.num_procs):
                    log_lik, kl_divergence = self.acmodel.vgae.backward(A_pred[j], adj_label, norm, weight_tensor)
                    # print(z.shape)
                    log_lik_lists += log_lik / self.num_procs
                    kl_divergence_lists += kl_divergence / self.num_procs
                
                loss_vgae = self.vage_log_lik_coef * log_lik_lists + self.vgae_kl_diverge_coef * kl_divergence_lists
                # starttime = datetime.datetime.now()
                self.vgae_optimizer.zero_grad()
                loss_vgae.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.vgae.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.vgae.parameters(), self.max_grad_norm)                
                self.vgae_optimizer.step()
                # endtime = datetime.datetime.now()
                # print('acmodel backward 耗时：', (endtime - starttime).seconds)
                                
                # import pdb
                # pdb.set_trace()
                self.log_lik[i] = log_lik_lists
                self.kl_diverge[i] = kl_divergence_lists
                # z : 64,1,64
                z_tensor = torch.mean(z, 1) # shape: self.num_procs, hidden 
                # print(z_tensor.shape)
                z_total_tensor_list.append(z_tensor)
                # endtime = datetime.datetime.now()
                # print('vgae 耗时：', (endtime - starttime))

                # print(z_tensor.shape)
                
            
            # import datetime
            # starttime = datetime.datetime.now()
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, z_tensor, self.memory * self.mask.unsqueeze(1)) # memeory 64*256  mask 64*1  --> {'dist':Categorical [64,7]; 'value' 64,7 'memory': 64,256 'ex..'}
        
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            # endtime = datetime.datetime.now()
            # print('acmodel 耗时：', (endtime - starttime))
            action = dist.sample() # 64,
            self.action_ = action

            obs, reward, done, env_info = self.env.step(action.cpu().numpy()) # action: ndarray(64,)  reward&done: tuple(64,)
            self.reward_ = torch.tensor(reward)
            self.preprocessed_obs_next_ = self.preprocess_obss(obs, device=self.device)
            
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            # import pdb
            # pdb.set_trace()
            self.obss[i] = self.obs
            self.obs = obs

            self.z_tensor[i] = z_tensor

            self.memories[i] = self.memory
            self.memory = memory

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

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        
        with torch.no_grad():
            z_tensor_next = torch.cat((z_tensor[1:, :], z_tensor[[-1], :]), dim=0)
            next_value = self.acmodel(preprocessed_obs, z_tensor_next, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[l][k]
                    for k in range(self.num_procs)
                    for l in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.z_tensors = self.z_tensors.transpose(0, 1).reshape(-1, *self.z_tensors.shape[2:])
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # import pdb
        # pdb.set_trace()
        
        exps.log_lik = self.log_lik.transpose(0, 1).reshape(-1)
        exps.kl_diverge = self.kl_diverge.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

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

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
