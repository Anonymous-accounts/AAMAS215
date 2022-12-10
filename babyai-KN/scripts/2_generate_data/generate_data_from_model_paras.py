#!/usr/bin/env python3

"""
Evaluate a trained model or bot
"""

import argparse
import gym
import time
import datetime

import os
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, evaluate

from babyai.arguments import ArgumentParser
# Parse arguments


"""save the buffer into npy"""
buffer_list = []
# storage_dir = '/home/username/data/BABYAI/'
root_ = '/data2/username_high/username/BABYAI/'
storage_dir = root_ + 'babyai-dyth-v1.1-and-baselines/scripts/3_pre_train_task/datasets_episode/'

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, default='',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="name of the demos file (REQUIRED or --demos-origin or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=10000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=int(1e9),
                    help="random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--contiguous-episodes", action="store_true", default=False,
                    help="Make sure episodes on which evaluation is done are contiguous")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")


def main(args, seed, episodes):
    # Set seed for all randomness sources
    utils.seed(seed)

    # Define agent

    env = gym.make(args.env)
    env.seed(seed)
    agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)
    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos
        episodes = len(agent.demos)

    # Evaluate
    if isinstance(agent, utils.DemoAgent):
        logs = evaluate_demo_agent(agent, episodes)
    elif isinstance(agent, utils.BotAgent) or args.contiguous_episodes:
        logs = evaluate(agent, env, episodes, False)
    else:
        logs = batch_evaluate(agent, args.env, seed, episodes)


    return logs



class ManyEnvs(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.done = [False] * len(self.envs)

    def seed(self, seeds):
        [env.seed(seed) for seed, env in zip(seeds, self.envs)]

    def reset(self):
        many_obs = [env.reset() for env in self.envs]
        self.done = [False] * len(self.envs)
        return many_obs

    def step(self, actions):
        self.results = [env.step(action) if not done else self.last_results[i]
                        for i, (env, action, done)
                        in enumerate(zip(self.envs, actions, self.done))]
        self.done = [result[2] for result in self.results]
        self.last_results = self.results
        return zip(*self.results)

    def render(self):
        raise NotImplementedError

# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False, pixel=False):
    # num_envs = min(256, episodes)
    num_envs = min(1, episodes)
    MAX_EPISODE_LEN = 0

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        if pixel:
            env = RGBImgPartialObsWrapper(env)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)

        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():
            action = agent.act_batch(many_obs)['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs_next, reward, done, _ = env.step(action)
            buffer_tuple = (many_obs, action.item(), reward, done[0], many_obs_next, cur_num_frames)
            buffer_list.append(buffer_tuple)
            many_obs = many_obs_next
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)
            
        if cur_num_frames > MAX_EPISODE_LEN:
            MAX_EPISODE_LEN = cur_num_frames
        print(i, MAX_EPISODE_LEN)

    return logs



if __name__ == "__main__":
    args = parser.parse_args()
    # args.model = root_ + 'models/BabyAI-OpenRedDoor-v0_ppo_bow_endpool_res_gru_mem_seed1_22-06-04-16-51-42_best'
    # args.env = 'BabyAI-GoToRedBall-v0'
    # args.model = '/data2/username_high/username/BABYAI/models/BabyAI-PutNext-v0_ppo_bow_endpool_res_gru_mem_seed1_22-06-14-05-03-55/'
    # args.env = 'BabyAI-PutNext-v0'
    assert_text = "ONE of --model or --demos-origin or --demos must be specified."
    assert int(args.model is None) + int(args.demos_origin is None) + int(args.demos is None) == 2, assert_text

    start_time = time.time()
    logs = main(args, args.seed, args.episodes)
    end_time = time.time()


    """ save the npy"""
    lst_npy = np.array(buffer_list)
    data_name = args.env + '_best_data'
    data_dir = os.path.join(storage_dir, "best_data_differ4", data_name)
    data_path = os.path.join(data_dir, "data_"+str(args.episodes)+".npy")
    utils.create_folders_if_necessary(data_path)
    np.save(data_path, lst_npy, allow_pickle=True)

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)

    if args.model is not None:
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    if args.model is not None:
        print("F {} | FPS {:.0f} | D {} | R:xsmM {:.3f} {:.3f} {:.3f} {:.3f} | S {:.3f} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      success_per_episode['mean'],
                      *num_frames_per_episode.values()))
    else:
        print("F {} | FPS {:.0f} | D {} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration, *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["num_frames_per_episode"])), key=lambda k: - logs["num_frames_per_episode"][k])

    n = args.worst_episodes_to_show
    if n > 0:
        print("{} worst episodes:".format(n))
        for i in indexes[:n]:
            if 'seed_per_episode' in logs:
                print(logs['seed_per_episode'][i])
            if args.model is not None:
                print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
            else:
                print("- episode {}: F={}".format(i, logs["num_frames_per_episode"][i]))
