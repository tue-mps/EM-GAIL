from collections import defaultdict
import gym
import numpy as np
import torch




class Sampler(object):
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, device, state_filter=None, seed=None):

        self.envs = np.asarray([gym.make(env_name) for i in range(n_trajectories)])

        i_seed = seed
        for env in self.envs:
            i_seed += 1
            env._max_episode_steps = trajectory_len
            env.seed(i_seed)

        self.policy = policy
        self.n_trajectories = n_trajectories
        self.trajectory_len = trajectory_len
        self.state_filter = state_filter
        self.device = device

    def sample_trajectories(self, deterministic, n, update_run_state, intention_idx):

        self.policy.eval()
        n_trajectories = self.n_trajectories
        envs = self.envs
        if n:
            n_trajectories = n
            envs = envs[:n]


        with torch.no_grad():
            memory = np.asarray([defaultdict(list) for i in range(n_trajectories)])
            done = [False] * n_trajectories
            for trajectory in memory:
                trajectory['done'] = False

            for env, trajectory in zip(envs, memory):
                state = torch.tensor(env.reset()).float()
                pure_state = state.clone()

                if self.state_filter:
                    state = self.state_filter(state, update_run_state)

                trajectory['states'].append(state)
                trajectory['pure_states'].append(pure_state)

            while not np.all(done):
                continue_mask = [i for i, trajectory in enumerate(memory) if not trajectory['done']]
                trajs_to_update = [trajectory for trajectory in memory if not trajectory['done']]
                continuing_envs = [env for i, env in enumerate(envs) if i in continue_mask]

                policy_input = torch.stack([trajectory['states'][-1].to(self.device)
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(policy_input)
                if deterministic:
                    actions = action_dists.mean
                else:
                    actions = action_dists.sample()

                for env, _, action, trajectory in zip(continuing_envs, policy_input, actions, trajs_to_update):
                    
                    state, reward, done, _ = env.step(action.cpu().numpy())
                    reward = reward[intention_idx]
                    reward_pred = reward


                    state = torch.tensor(state).float()
                    pure_state = state.clone()
                    reward = torch.tensor(reward, dtype=torch.float)
                    reward_pred = torch.tensor(reward_pred, dtype=torch.float)

                    if self.state_filter:
                        state = self.state_filter(state, update_run_state)

                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward_pred)
                    trajectory['rewards_true'].append(reward)
                    trajectory['done'] = done

                    if not done:
                        trajectory['states'].append(state)
                        trajectory['pure_states'].append(pure_state)

                done = [trajectory['done'] for trajectory in memory]

        return memory
