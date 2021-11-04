import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.running_mean_std import RunningMeanStd

class Discriminator(nn.Module):
    def __init__(self, intention_dim, state_dim, action_dim, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.intention_dim = intention_dim
        self.input_dim = state_dim + action_dim

        self.disc1 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            )
        self.disc2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            )

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def forward(self, x):
        sq = False
        if len(x.size()) == 1:
            sq = True
            x = x.unsqueeze(0)

        intention = torch.narrow(x, -1, 0, self.intention_dim)
        state_action = torch.narrow(x, -1, self.intention_dim, self.input_dim)
        l = len(x.size())

        out = []
        out.append(self.disc1(state_action.clone()))
        out.append(self.disc2(state_action.clone()))
        out = torch.stack(out, dim=l-1)
        intention = intention.unsqueeze(l).repeat(1,1,out.size(-1))
        out = torch.sum(out*intention, dim=l-1)

        if sq:
            out = out.squeeze(0)

        return out

class EMGail(nn.Module):
    def __init__(self, intention_dim, state_dim, action_dim, device, gail_epochs=1, hidden_dim=128):
        super(EMGail, self).__init__()

        self.device = device
        self.gail_epochs = gail_epochs

        self.discrim = Discriminator(intention_dim, state_dim, action_dim).to(device)


        self.optimizer_discrim = torch.optim.Adam(self.discrim.parameters())

        self.posterior = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, intention_dim)).to(device)


        self.optimizer_posterior = torch.optim.Adam(self.posterior.parameters())

    def update(self, expert_dataset, expert_loader, policy_loader, state_filter):

        posterior_loss, expert_intentions, cluster_pred = self.E_update(expert_dataset, policy_loader, state_filter)
        expert_dataset.intentions = expert_intentions
        discrim_loss = self.M_update(expert_loader, policy_loader, state_filter)
        return posterior_loss, cluster_pred, discrim_loss


    def M_update(self, expert_loader, policy_loader, state_filter):

        losses_discrim = []
        for _ in range(self.gail_epochs):

            
            for expert_batch, policy_batch in zip(expert_loader, policy_loader):

                policy_intention, policy_state, policy_action, _ = policy_batch
                policy_intention = policy_intention.to(self.device)
                policy_state = policy_state.to(self.device)
                policy_action = policy_action.to(self.device)

                expert_intention, expert_state, expert_action, _  = expert_batch
                expert_state = state_filter(expert_state.cpu(), False)
                expert_intention = expert_intention.to(self.device)
                expert_state = expert_state.to(self.device)
                expert_action = expert_action.to(self.device)

                # Discriminator
                policy_d = self.discrim(
                    torch.cat([policy_intention, policy_state, policy_action], dim=-1))
                policy_loss = F.binary_cross_entropy(
                    policy_d,
                    torch.zeros(policy_d.size()).to(self.device))

                expert_d = self.discrim(
                    torch.cat([expert_intention, expert_state, expert_action], dim=-1))
                expert_loss = F.binary_cross_entropy(
                    expert_d,
                    torch.ones(expert_d.size()).to(self.device))

             

                discrim_loss = expert_loss + policy_loss
                losses_discrim.append(discrim_loss.item())
                self.optimizer_discrim.zero_grad()
                discrim_loss.backward()
                self.optimizer_discrim.step()


        discrim_loss = np.mean(losses_discrim)
        
        return discrim_loss

    def E_update(self, expert_dataset, policy_loader, state_filter):
        

        losses_posterior = []
        for _ in range(self.gail_epochs):

            for policy_batch in policy_loader:

                _, policy_state, policy_action, policy_intention_idx = policy_batch
                policy_state = policy_state.to(self.device)
                policy_action = policy_action.to(self.device)
                policy_intention_idx = policy_intention_idx.to(self.device)

                policy_p = self.posterior(
                    torch.cat([policy_state, policy_action], dim=-1))
                posterior_loss = F.cross_entropy(policy_p, policy_intention_idx)
                losses_posterior.append(posterior_loss.item())
                self.optimizer_posterior.zero_grad()
                posterior_loss.backward()
                self.optimizer_posterior.step()

        posterior_loss = np.mean(losses_posterior)

        expert_states = expert_dataset.states
        expert_states = state_filter(expert_states.cpu(), False).to(self.device)
        expert_actions = expert_dataset.actions.to(self.device)
        
        with torch.no_grad():
            expert_intentions = self.posterior(torch.cat([expert_states, expert_actions], dim=-1))
            expert_intentions = torch.softmax(expert_intentions, dim=-1)
            cluster_pred = torch.argmax(expert_intentions, dim=-1).cpu()



        
        return posterior_loss, expert_intentions, cluster_pred


