import torch
class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, all_samples, n_intentions):

        all_states = []
        all_actions = []
        cluster_true = []
        all_intentions = []
        all_avg_rewards = []
        for intention_idx in range(n_intentions):
            states = torch.cat([torch.stack(trajectory['pure_states']) for trajectory in all_samples[intention_idx]])
            actions = torch.cat([torch.stack(trajectory['actions']) for trajectory in all_samples[intention_idx]])
            all_states.append(states)
            all_actions.append(actions)
            curr_clus = torch.zeros(states.size(0), dtype=torch.int)
            curr_clus[:] = intention_idx
            cluster_true.append(curr_clus)
            intentions = torch.zeros(states.size(0), n_intentions)
            intentions[:,intention_idx] = 1
            all_intentions.append(intentions)

            avg_reward = [torch.stack(trajectory['rewards_true'], dim=0).sum(0) for trajectory in all_samples[intention_idx]]
            avg_reward = torch.stack(avg_reward, dim=0).mean(dim=0)
            all_avg_rewards.append(avg_reward)
            
        self.rewards_true = all_avg_rewards
        self.cluster_true = torch.cat(cluster_true)
        self.states = torch.cat(all_states)
        self.actions = torch.cat(all_actions)
        self.intention_idxs = torch.cat(cluster_true)
        self.intentions = torch.cat(all_intentions)

       

            
            
    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, i):
        intention = self.intentions[i]
        state = self.states[i]
        action = self.actions[i]
        intention_idx = self.intention_idxs[i]

        return [intention, state, action, intention_idx]