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

    def sample_trajectories(self, intention, intention_idx, deterministic, n, update_run_state, custom_reward):

        self.policy.eval()
        n_trajectories = self.n_trajectories
        envs = self.envs
        if n:
            n_trajectories = n
            envs = envs[:n]

        intention = torch.tensor(intention).float()
        intention_idx = torch.tensor(intention_idx)

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
                trajectory['intention'].append(intention)
                trajectory['intention_idx'].append(intention_idx)

            while not np.all(done):
                continue_mask = [i for i, trajectory in enumerate(memory) if not trajectory['done']]
                trajs_to_update = [trajectory for trajectory in memory if not trajectory['done']]
                continuing_envs = [env for i, env in enumerate(envs) if i in continue_mask]

                policy_input_states = torch.stack([trajectory['states'][-1].to(self.device)
                                            for trajectory in trajs_to_update])
                policy_input_intentions = torch.stack([trajectory['intention'][-1].to(self.device)
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(torch.cat((policy_input_intentions, policy_input_states), dim=-1))
                if deterministic:
                    actions = action_dists.mean
                else:
                    actions = action_dists.sample()

                for env, prev_state, action, trajectory in zip(continuing_envs, policy_input_states, actions, trajs_to_update):
                    
                    state, reward, done, info = env.step(action.cpu().numpy())

                    reward_pred = custom_reward(intention_idx, intention.to(self.device), prev_state, action)


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
                        trajectory['intention'].append(intention)
                        trajectory['intention_idx'].append(intention_idx)

                done = [trajectory['done'] for trajectory in memory]
            
        return memory
