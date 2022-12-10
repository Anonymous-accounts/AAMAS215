#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model_ours_v2 import ACModel
from babyai.evaluate_ours_v2 import batch_evaluate
from babyai.utils.agent_ours_v2 import ModelAgent
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from babyai.models.encoder import *


# Parse arguments
parser = ArgumentParser()
# parser.add_argument("--env", default='ppo',
#                     help="algorithm to use (default: ppo)")
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()


args.env = 'BabyAI-BossLevel-v0'
args.tb = True
args.model = 'ppo'
# args.procs = 2
### TODO: 目前该版本出现的问题：memory可能是混乱的，因为不断变换model，另外model不能训练

#### load subtask enocoder models

args.model_path = '/data2/username_high/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_4_tasks_new/[09-07]11.26.22train_loader_bs_64_corro_others_neg_/models/encoder_epoch_740000.pt'
#args.model_path = '/home/username/BABYAI/babyai-dyth-v1.1-and-baselines-CORRO/scripts/3_pre_train_task/train_task_v9_recon_n_enc_1_dec/logs_models_v9_4_tasks_new/[09-07]11.26.22train_loader_bs_64_corro_others_neg_/models/encoder_epoch_740000.pt'
pre_subtask_encoder = torch.load(args.model_path).cuda()
#### define the query encoder
args.obs_dim, args.n_actions, args.normalize_z, args.hidden_size_lst = 147, 7, True, [32, 16, 8]  ### 
args.task_embedding_size = 5
query_encoder = QueryMLPEncoder(
                hidden_size_lst=args.hidden_size_lst, # 256
                action_size=args.n_actions, # 7
                state_size=args.obs_dim, # 147
                normalize_z=args.normalize_z, # True
                task_embedding_size=args.task_embedding_size
        	).cuda()  ### 包含一个encoder，映射到task_emb，query 预训练的子任务encoder；； 以及一个decoder，循环生成下一个状态

##### define att network
args.n_embd, args.n_head, args.attn_pdrop, args.resid_pdrop = args.task_embedding_size, 1, 0.1, 0.1
args.n_embd_q, args.n_embd_kv = args.task_embedding_size, args.task_embedding_size
att = CausalSelfAttention(args).cuda()

args.load_ac_models = True

utils.seed(args.seed)

# Generate environments
envs = []
use_pixel = 'pixel' in args.arch
for i in range(args.procs):  ## 64
    env = gym.make(args.env)
    if use_pixel:
        env = RGBImgPartialObsWrapper(env)
    env.seed(100 * args.seed + i)
    envs.append(env)

# Define model name
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
model_name_parts = {
    'env': args.env,
    'algo': args.algo,
    'arch': args.arch,
    'instr': instr,
    'mem': mem,
    'seed': args.seed,
    'info': '',
    'coef': '',
    'suffix': suffix}
default_model_name = "{env}_{algo}_Ours_v2_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name

# import pdb
# pdb.set_trace()
# args.model = args.model.format(**model_name_parts) if args.model else default_model_name
args.model = default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

# Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    # elif args.load_ac_models:
    #     acmodels = 
    else:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space, # {'image': 147, 'instr': 100}  # Discrete(7)
                          args.image_dim, args.memory_dim, args.instr_dim,   # 128 128 128
                          not args.no_instr, args.instr_arch, not args.no_mem, args.arch)  # True, 'gru', True, 'bow_endpool_res'

obss_preprocessor.vocab.save()
utils.save_model(acmodel, args.model)

if torch.cuda.is_available():
    acmodel.cuda()

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo_Ours_v2(envs, acmodel, (query_encoder, pre_subtask_encoder, att), args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                             args.gae_lambda,
                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                             args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                             reshape_reward)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}

# Define logger and Tensorboard writer and CSV writer

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(utils.get_log_dir(args.model))
csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# Train model

total_start_time = time.time()
best_success_rate = 0
best_mean_return = 0
test_env_name = args.env
while status['num_frames'] < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()
    
    import gc
    gc.collect() 
    torch.cuda.empty_cache()

    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1

    # Print logs

    if status['i'] % args.log_interval == 0:
        
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

        csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        
        # utils.save_model(acmodel, args.model + '_best')
        # obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
        # Testing the model before saving
        agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
        agent.model = acmodel
        agent.model.eval()
        query_encoder.eval()
        pre_subtask_encoder.eval()
        att.eval()
        
        # query_encoder, pre_subtask_encoder, att
        encoder_tuple = query_encoder, pre_subtask_encoder, att
        logs = batch_evaluate(agent, encoder_tuple, test_env_name, args.val_seed, args.val_episodes, pixel=use_pixel)
        agent.model.train()
        query_encoder.train()
        pre_subtask_encoder.train()
        att.train()
        
        mean_return = np.mean(logs["return_per_episode"])
        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        save_model = False
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            save_model = True
        elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
            best_mean_return = mean_return
            save_model = True
        if save_model:
            utils.save_model(acmodel, args.model + '_best')
            obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
            logger.info("Return {: .2f}; best model is saved".format(mean_return))
        else:
            logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
