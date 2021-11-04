
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from utils import cluster_acc, compute_AERD
from PIL import Image



class Agent:

    def __init__(self, rl_net, sampler, policy, env, device, irl_net, expert_dataset, irl_type, random_policy, n_intentions):
        self.sampler = sampler
        self.rl_net = rl_net
        self.last_data = None
        self.policy = policy
        self.env = env
        self.device = device
        self.irl_net = irl_net
        self.irl_type = irl_type
        self.random_policy = random_policy
        self.n_intentions = n_intentions
        self.expert_dataset = expert_dataset
        self.expert_loader = DataLoader(
            dataset=self.expert_dataset,
            batch_size=500,
            shuffle=True)
        self.policy_dataset = deepcopy(expert_dataset)
        self.policy_loader = DataLoader(
            dataset=self.policy_dataset,
            batch_size=500,
            shuffle=True)
        self.cluster_true = expert_dataset.cluster_true

    def custom_reward(self, intention_idx, intention, state, action, coeff=0.4):

        with torch.no_grad():
            
            d = self.irl_net.discrim(torch.cat([intention, state, action], dim=-1))
            reward_discrim =  - (1 - d + 1e-20).log()
            reward_pred = reward_discrim.item()

           


        return reward_pred
    
    def update_dataset(self, all_samples):

        all_intentions = []
        all_states = []
        all_actions = []
        all_intention_idxs = []

        for i in range(len(all_samples)):
            intentions = torch.cat([torch.stack(trajectory['intention']) for trajectory in all_samples[i]])
            states = torch.cat([torch.stack(trajectory['states']) for trajectory in all_samples[i]])
            actions = torch.cat([torch.stack(trajectory['actions']) for trajectory in all_samples[i]])
            intention_idx = torch.cat([torch.stack(trajectory['intention_idx']) for trajectory in all_samples[i]])
            all_intentions.append(intentions)
            all_states.append(states)
            all_actions.append(actions)
            all_intention_idxs.append(intention_idx)
        self.policy_dataset.intentions = torch.cat(all_intentions, dim=0)
        self.policy_dataset.states = torch.cat(all_states, dim=0)
        self.policy_dataset.actions = torch.cat(all_actions, dim=0)
        self.policy_dataset.intention_idxs = torch.cat(all_intention_idxs, dim=0)


    def collect_samples(self, intention, intention_idx, deterministic=False, n=None, update_run_state=True):

        sampler = self.sampler

        log = dict()
        samples = sampler.sample_trajectories(intention, intention_idx, deterministic, n, update_run_state, self.custom_reward)

        avg_reward = [torch.stack(trajectory['rewards_true'], dim=0).sum(0) for trajectory in samples]
        avg_reward = torch.stack(avg_reward, dim=0).mean(dim=0)[:self.n_intentions]
        log['avg_reward'] = np.array(avg_reward)
        log['avg_length'] = np.mean([len(trajectory['rewards_true']) for trajectory in samples])

        return log, samples 
    
    def update(self, all_demons, all_samples):

        
        self.update_dataset(all_samples) 
        rl_net = self.rl_net
        last_data = self.last_data
        expert_loader = self.expert_loader
        policy_loader = self.policy_loader
        expert_dataset = self.expert_dataset
        
        
        irl_net = self.irl_net
        state_filter = self.sampler.state_filter

        posterior_loss, cluster_pred, discrim_loss = irl_net.update(expert_dataset, expert_loader, policy_loader, state_filter)

        cluster_accuracy = cluster_acc(cluster_pred, self.cluster_true)


        # rl_net
        all_intentions = []
        all_states = []
        all_actions = []
        all_q_vals = []
        all_advantages = []
        all_states_with_time = []
        for i in range(len(all_samples)):

            intentions, states, actions, q_vals = rl_net.unroll_samples(all_samples[i])
            advantages, states_with_time = rl_net.get_advantages(all_samples[i])
            advantages -= torch.mean(advantages)
            advantages /= torch.std(advantages)

            all_intentions.append(intentions)
            all_states.append(states)
            all_actions.append(actions)
            all_q_vals.append(q_vals)
            all_advantages.append(advantages)
            all_states_with_time.append(states_with_time)
        intentions = torch.cat(all_intentions, dim=0)
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        q_vals = torch.cat(all_q_vals, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        states_with_time = torch.cat(all_states_with_time, dim=0)

        rl_net.update_policy(intentions, states, actions, advantages)

        if last_data is not None:
            last_q, last_intentions, last_states_with_time = last_data
            rl_net.update_value_fun(
                torch.cat([intentions, last_intentions]),
                torch.cat([states_with_time, last_states_with_time]), 
                torch.cat([q_vals, last_q])
            )
        else:
            rl_net.update_value_fun(intentions, states_with_time, q_vals)

        self.last_data = [q_vals, intentions, states_with_time]

        return discrim_loss, posterior_loss, cluster_accuracy

    def test_policy(self, intention, img_path, save=False):
        
        n = 1
        if save:
            n = 1
        policy = self.policy
        env = self.env
        device = self.device
        state_filter = self.sampler.state_filter
        log = dict()
        all_rewards = []
        all_lengths = []
        intention = torch.tensor(intention).float()
        intention = intention.to(device)


        for _ in range(n):
            policy.eval()
            state = env.reset()
            done = False
            rewards = []
            t = 0
            
            while not done:
                state = torch.tensor(state).float()
                
                
                if state_filter:
                    state = state_filter(state, update=False)
                    
                state = state.to(device)
                
                with torch.no_grad():
                    action_dist = policy(torch.cat((intention,state), dim=-1))
                    action = action_dist.mean.cpu()

                state, reward, done, _ = env.step(action.numpy())
                rewards.append(reward)
                
                if save:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save(img_path+'_'+str(t)+'.jpg')
                else:
                    env.render()
                t+=1  
            all_rewards.append(np.sum(np.stack(rewards, axis=0) , axis = 0))
            all_lengths.append(len(rewards))

        log['avg_reward'] = np.mean(np.stack(all_rewards, axis=0), axis=0)
        log['avg_length'] = np.mean(all_lengths)

        return log

    def test_random_policy(self):
        
        policy = self.random_policy
        env = self.env
        device = self.device
        n_intentions = self.n_intentions
        log = dict()
        all_rewards = []
        intention = torch.zeros(n_intentions, device=device)


        for _ in range(5):
            policy.eval()
            state = env.reset()
            done = False
            rewards = []
            
            while not done:
                state = torch.tensor(state).float().to(device)
                
                with torch.no_grad():
                    action_dist = policy(torch.cat((intention,state), dim=-1))
                    action = action_dist.mean.cpu()

                state, reward, done, _ = env.step(action.numpy())
                rewards.append(reward)
            all_rewards.append(np.sum(np.stack(rewards, axis=0) , axis = 0))

        all_rewards = np.mean(np.stack(all_rewards, axis=0), axis=0)[:n_intentions]
        rewards = []
        for _ in range(n_intentions):
            rewards.append(all_rewards)

        max_AERD, _ = compute_AERD(rewards, self.expert_dataset.rewards_true)
        min_AERD = 0

        return max_AERD, min_AERD






        
