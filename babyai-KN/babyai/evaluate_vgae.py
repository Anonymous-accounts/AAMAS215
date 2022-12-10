import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
import torch



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


def trans_image(image):
    return torch.transpose(torch.transpose(image, 1, 3), 2, 3)

def list_onehot(actions: list, n: int, device) -> torch.Tensor:
    """
    列表动作值转 onehot
    actions: 动作列表
    n: 动作总个数
    """
    result = []
    for action in actions:
        result.append([int(k == action) for k in range(n)])
    result = torch.tensor(result, dtype=torch.float).to(device)
    return result

def vage_processing_inputs(data_x, num_procs, device): # 0, 63
    """
    concatenate 转换成 vgae的输入
    data_x: tuple(s,a,r,s')
    """
    n_actions = 7
    data_x_1 = trans_image(data_x[0].image).reshape(num_procs, -1)
    data_x_2 = list_onehot(data_x[1], n_actions, device) # action
    data_x_3 = data_x[2][:, None].float().to(device)  # rewards
    data_x_4 = trans_image(data_x[3].image).reshape(num_procs, -1)
    data_x_1234 = torch.cat((data_x_1, data_x_2, data_x_3, data_x_4), dim=1)
    return data_x_1234

def generate_seq_adj(EPS_TIMESTEP):
        adj_zeros_numpy = np.zeros((EPS_TIMESTEP, EPS_TIMESTEP))
        if EPS_TIMESTEP == 1:
            adj_zeros_numpy[0][0] = 1
        for i in range(0, EPS_TIMESTEP-1):
            adj_zeros_numpy[i][i+1] = 1
        return adj_zeros_numpy

# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False, pixel=False):
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

    for i in range((episodes + num_envs - 1) // num_envs):
        # import pdb
        # pdb.set_trace()
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
        
        vgae_inputs_list = []
        preprocessed_obs_0 = agent.obss_preprocessor(many_obs, agent.device)
        preprocessed_obs_next_0 = preprocessed_obs_0
        num_batch = preprocessed_obs_0.instr.shape[0]
        action_0 = torch.zeros([num_batch]).to(agent.device)
        reward_0 = torch.zeros([num_batch]).to(agent.device)
        vgae_inputs_init = vage_processing_inputs((preprocessed_obs_0, action_0, reward_0, preprocessed_obs_next_0), num_batch, agent.device)
        vgae_inputs_list.append(vgae_inputs_init)
        # count = 0
        while (num_frames == 0).any():
            #### 生成vgae的结点，边的输入输出
            vgae_inputs_tensor = torch.stack(vgae_inputs_list, dim=1) # torch.Size([64, 1, 302])
            EPS_TIMESTEP = len(vgae_inputs_list)
            adj_ = generate_seq_adj(EPS_TIMESTEP)
            adj_norm, adj_label, norm, weight_tensor = agent.model.vgae.data_transfer(adj_)
            adj_lists = [agent.model.vgae.transfer_edge_index(adj_norm.to_dense()) for _ in range(num_batch)]
            z, A_pred = agent.model.vgae.forward_batch(vgae_inputs_tensor, adj_lists)
            z_tensor = torch.mean(z, 1) # shape: self.num_procs, hidden 
            
            action_dict, preprocessed_obs = agent.act_batch(many_obs, z_tensor)
            action = action_dict['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs, reward, done, _ = env.step(action)
            ### precess obs
            preprocessed_obs_next = agent.obss_preprocessor(many_obs, agent.device)

            vgae_inputs_ = vage_processing_inputs((preprocessed_obs, action, torch.tensor(reward), preprocessed_obs_next), num_batch, agent.device)
            vgae_inputs_list.append(vgae_inputs_)

            # many_obs = many_obs_next
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

    return logs
