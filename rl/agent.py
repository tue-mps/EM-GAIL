
import numpy as np
import torch



class Agent:

    def __init__(self, trpo, sampler, policy, env, intention_idx, device):
        self.sampler = sampler
        self.trpo = trpo
        self.last_data = None
        self.policy = policy
        self.env = env
        self.intention_idx = intention_idx
        self.device = device


    def collect_samples(self, deterministic=False, n=None, update_run_state=True):

        sampler = self.sampler

        log = dict()
        
        samples = sampler.sample_trajectories(deterministic, n, update_run_state, self.intention_idx)

        log['avg_reward'] = np.mean([np.sum(trajectory['rewards_true']) for trajectory in samples])
        log['avg_length'] = np.mean([len(trajectory['rewards_true']) for trajectory in samples])

        return log, samples 
    
    def update(self, samples):

        

        trpo = self.trpo
        last_data = self.last_data
    
        states, actions, _, q_vals = trpo.unroll_samples(samples)
        advantages, states_with_time = trpo.get_advantages(samples)
        advantages -= torch.mean(advantages)
        advantages /= torch.std(advantages)

        trpo.update_policy(states, actions, advantages)

        if last_data is not None:
            last_q, last_states = last_data
            trpo.update_value_fun(torch.cat([states_with_time, last_states]), torch.cat([q_vals, last_q]))
        else:
            trpo.update_value_fun(states_with_time, q_vals)

        self.last_data = [q_vals, states_with_time]

    def test_policy(self):

        policy = self.policy
        env = self.env
        device = self.device
        state_filter = self.sampler.state_filter
        log = dict()
        all_rewards = []
        all_lengths = []


        for i in range(1):
            policy.eval()
            state = env.reset()
            done = False
            rewards = []
            
            while not done:
                state = torch.tensor(state).float()
                if state_filter:
                    state = state_filter(state)
                    
                state = state.to(device)
                with torch.no_grad():
                    action_dist = policy(state)
                    action = action_dist.mean.cpu()

                state, reward, done, _ = env.step(action.numpy())
                rewards.append(reward[self.intention_idx])
                env.render()
            all_rewards.append(np.sum(rewards))
            all_lengths.append(len(rewards))

        log['avg_reward'] = np.mean(all_rewards)
        log['avg_length'] = np.mean(all_lengths)

        return log






        
