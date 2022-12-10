import numpy
import torch
import torch.nn.functional as F

from torchkit import pytorch_utils as ptu
from torch.distributions import Categorical

from babyai.rl.algos.base_ours_v2 import BaseAlgo_Ours_v2


class PPOAlgo_Ours_v2(BaseAlgo_Ours_v2):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, encoder_tuple, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, encoder_tuple, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_actions = 7

        self.query_encoder = encoder_tuple[0]
        self.pre_subtask_encoder = encoder_tuple[1]
        self.att = encoder_tuple[2]

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.extra_parameteres = list(self.query_encoder.parameters()) + list(self.att.parameters())
        self.optimizer_query = torch.optim.Adam(self.extra_parameteres, lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        import datetime
        starttime = datetime.datetime.now()
        exps, logs = self.collect_experiences()
        endtime = datetime.datetime.now()
        # print('收集数据耗时：', (endtime - starttime).seconds)
        
        import gc
        gc.collect() 
        torch.cuda.empty_cache()
        
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = [0]
            log_values = [0]
            log_policy_losses = [0]
            log_value_losses = [0]
            log_grad_norms = [0]

            log_losses = [0]

            log_loss_extras, log_loss_mses, log_loss_atts = [], [], []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                batch_loss_extra = 0
                batch_loss_mse = 0
                batch_loss_att = 0


                # Initialize memory
                memory = exps.memory[inds]

                for i in range(self.recurrence-1):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    # model_results = self.acmodel.forward_train(sb.obs, memory * sb.mask, sb.task_embs)
                    # dist = model_results['dist']
                    # value = model_results['value']
                    # memory = model_results['memory']
                    # extra_predictions = model_results['extra_predictions']

                    # entropy = dist.entropy().mean()

                    # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    # surr1 = ratio * sb.advantage
                    # surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    # policy_loss = -torch.min(surr1, surr2).mean()

                    # value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    # surr1 = (value - sb.returnn).pow(2)
                    # surr2 = (value_clipped - sb.returnn).pow(2)
                    # value_loss = torch.max(surr1, surr2).mean()

                    # loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # # self.optimizer.zero_grad()
                    # # batch_loss_extra.backward()
                    # # grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.extra_parameteres if p.grad is not None) ** 0.5
                    # # torch.nn.utils.clip_grad_norm_(self.extra_parameteres, self.max_grad_norm)
                    # # self.optimizer.step()


                    # # Update batch values

                    # batch_entropy += entropy.item()
                    # batch_value += value.mean().item()
                    # batch_policy_loss += policy_loss.item()
                    # batch_value_loss += value_loss.item()
                    # batch_loss += loss

                    ######## update e
                    flatten_obs = torch.transpose(torch.transpose(sb.obs.image, 1, 3), 2, 3).reshape(len(inds), -1) # 64.147
                    last_act = torch.zeros(len(inds), self.n_actions).cuda().scatter_(1, exps[inds+i-1].action[:,None].long(), 1) # 2.7
                    last_r = exps[inds+i-1].returnn[:,None]
                    query_emb = self.query_encoder(flatten_obs, last_act, last_r)   # torch.Size([1, 5])
                    pred_obs = self.query_encoder.pred(query_emb)  # 2.147
                    # MSE loss
                    loss_mse_fn = torch.nn.MSELoss(reduction='mean')
                    flatten_obs_next = torch.transpose(torch.transpose(exps[inds+i+1].obs.image, 1, 3), 2, 3).reshape(len(inds), -1) # 64.147
                    loss_mse = loss_mse_fn(pred_obs, flatten_obs_next)
                    batch_loss_extra += loss_mse
                    # print(loss_mse)

                    pre_subtask_embs = self.pre_subtask_encoder(flatten_obs, last_act, last_r) # torch.Size([4, 4, 5])
                    query_probs = self.att.att_prob(query_emb[:,None].detach(), pre_subtask_embs) # 4.4.1

                    query_probs = query_probs + 10e-7

                    # print('----------------------------', query_probs)
                    sample_dist = Categorical(query_probs[:,:,0])

                    ratio = torch.exp(sample_dist.log_prob(sb.nodes) - sb.edges.detach())
                    att_actor_loss = (ratio * sb.advantage).mean()

                    
                    batch_loss_mse += loss_mse.item()
                    batch_loss_att += att_actor_loss.item()
                    batch_loss_extra += att_actor_loss
                    
                    # Update memories for next epoch
                    del last_act

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                # batch_entropy /= self.recurrence
                # batch_value /= self.recurrence
                # batch_policy_loss /= self.recurrence
                # batch_value_loss /= self.recurrence
                # batch_loss /= self.recurrence
                batch_loss_extra /= self.recurrence
                batch_loss_mse /= self.recurrence
                batch_loss_att /= self.recurrence

                # Update actor-critic

                # self.optimizer.zero_grad()
                # batch_loss.backward()
                # grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                # torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                # self.optimizer.step()

                # Update Query_Encoder

                self.optimizer_query.zero_grad()
                batch_loss_extra.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.extra_parameteres if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.extra_parameteres, self.max_grad_norm)
                self.optimizer_query.step()


                # Update log values

                # log_entropies.append(batch_entropy)
                # log_values.append(batch_value)
                # log_policy_losses.append(batch_policy_loss)
                # log_value_losses.append(batch_value_loss)
                # log_grad_norms.append(grad_norm.item())
                # log_losses.append(batch_loss.item())

                log_loss_extras.append(batch_loss_extra.item())
                log_loss_mses.append(batch_loss_mse)
                log_loss_atts.append(batch_loss_att)

                # print('back once')

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)
        logs["loss_extras"] = numpy.mean(log_loss_extras)
        logs["loss_mse"] = numpy.mean(log_loss_mses)
        logs["loss_att"] = numpy.mean(log_loss_atts)
        
        import gc
        gc.collect() 
        torch.cuda.empty_cache()
        
        del exps
        # exps.clear()

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
