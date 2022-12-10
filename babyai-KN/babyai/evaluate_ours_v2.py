import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

import torch

from babyai.models.tree import TaskTree

# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes, model_agent=True, offsets=None):
    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    if offsets:
        count = 0

    for i in range(episodes):
        if offsets:
            # Ensuring test on seed offsets that generated successful demonstrations
            while count != offsets[i]:
                obs = env.reset()
                count += 1

        obs = env.reset()
        agent.on_reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward


        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    if model_agent:
        agent.model.train()
    return logs


def evaluate_demo_agent(agent, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    number_of_demos = len(agent.demos)

    for demo_id in range(min(number_of_demos, episodes)):
        logs["num_frames_per_episode"].append(len(agent.demos[demo_id]))

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
def batch_evaluate(agent, encoder_tuple, env_name, seed, episodes, return_obss_actions=False, pixel=False):
    query_encoder, pre_subtask_encoder, att = encoder_tuple
    num_envs = min(256, episodes)

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
    
    N, n_actions = 3, 7
    num_procs = num_envs
    task_tree = TaskTree(num_procs, N)

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
                
        reconstrction_flag = True
        last_act = torch.zeros(num_procs, n_actions).cuda()
        last_r = torch.zeros(num_procs, 1).cuda()
        while (num_frames == 0).any():
            
            ####### begin task tree         
            if task_tree.tree_visited[0][-1]: #### 都访问完了
                reconstrction_flag = True
            if reconstrction_flag:
                preprocessed_obs = agent.obss_preprocessor(many_obs, device=agent.device)
                task_tree.init()
                if agent.memory is None:
                    agent.memory = torch.zeros(len(many_obs), agent.model.memory_size, device=agent.device)
                task_tree.construct_tree(query_encoder, pre_subtask_encoder, att, agent.model, preprocessed_obs, agent.memory)
            task_tree.last_act, task_tree.last_r = last_act, last_r
            ####### end task tree ,, next 15 lines continue
            
            
            action = agent.act_batch(many_obs, task_tree)['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True
            
            
            ####### begin task tree
            task_tree.achor += 1
            last_act = torch.zeros(num_procs, n_actions).cuda().scatter_(1, action[:,None], 1) # 2.7
            last_r = torch.tensor(reward).cuda()[:,None]
            ####### end task tree
            
        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    del task_tree
    return logs
