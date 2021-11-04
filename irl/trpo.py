import torch
from torch.nn import MSELoss
from torch.optim import LBFGS

from utils import cg_solver, mean_kl_first_fixed, get_Hvp_fun, line_search, apply_update, flat_grad, get_device, get_flat_params


class TRPO:

    def __init__(self, policy, value_fun, simulator, device, max_kl_div=0.01, max_value_step=0.01,
                vf_iters=1, vf_l2_reg_coef=1e-3, discount=0.995, lam=0.98, cg_damping=1e-3,
                cg_max_iters=10, line_search_coef=0.9, line_search_max_iter=10,
                line_search_accept_ratio=0.1):


        self.policy = policy
        self.value_fun = value_fun
        self.simulator = simulator
        self.max_kl_div = max_kl_div
        self.max_value_step = max_value_step
        self.vf_iters = vf_iters
        self.vf_l2_reg_coef = vf_l2_reg_coef
        self.discount = discount
        self.lam = lam
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.line_search_coef = line_search_coef
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio
        self.mse_loss = MSELoss(reduction='mean')
        self.value_optimizer = LBFGS(self.value_fun.parameters(), lr=max_value_step, max_iter=25)
        self.episode_num = 0
        self.device = device



    def unroll_samples(self, samples):
        q_vals = []

        for trajectory in samples:
            rewards = torch.tensor(trajectory['rewards'])
            reverse = torch.arange(rewards.size(0) - 1, -1, -1)
            discount_pows = torch.pow(self.discount, torch.arange(0, rewards.size(0)).float())
            discounted_rewards = rewards * discount_pows
            disc_reward_sums = torch.cumsum(discounted_rewards[reverse], dim=-1)[reverse]
            trajectory_q_vals = disc_reward_sums / discount_pows
            q_vals.append(trajectory_q_vals)

        
        intentions = torch.cat([torch.stack(trajectory['intention']) for trajectory in samples])
        states = torch.cat([torch.stack(trajectory['states']) for trajectory in samples])
        actions = torch.cat([torch.stack(trajectory['actions']) for trajectory in samples])
        q_vals = torch.cat(q_vals)

        return intentions, states, actions, q_vals

    def get_advantages(self, samples):
        advantages = []
        states_with_time = []
        T = self.simulator.trajectory_len

        for trajectory in samples:
            time = torch.arange(0, len(trajectory['rewards'])).unsqueeze(1).float() / T
            intentions = torch.stack(trajectory['intention'])
            intentions = intentions.to(self.device)
            states = torch.stack(trajectory['states'])
            states = torch.cat([states, time], dim=-1)
            states = states.to(self.device)
            states_with_time.append(states.cpu())
            rewards = torch.tensor(trajectory['rewards'])

            with torch.no_grad():
                state_values = self.value_fun(torch.cat((intentions, states), dim=-1))
            state_values = state_values.view(-1)
            state_values = state_values.cpu()
            state_values_next = torch.cat([state_values[1:], torch.tensor([0.0])])

            td_residuals = rewards + self.discount * state_values_next - state_values
            reverse = torch.arange(rewards.size(0) - 1, -1, -1)
            discount_pows = torch.pow(self.discount * self.lam, torch.arange(0, rewards.size(0)).float())
            discounted_residuals = td_residuals * discount_pows
            disc_res_sums = torch.cumsum(discounted_residuals[reverse], dim=-1)[reverse]
            trajectory_advs = disc_res_sums / discount_pows
            advantages.append(trajectory_advs)

        advantages = torch.cat(advantages)

        states_with_time = torch.cat(states_with_time)

        return advantages, states_with_time

    def update_value_fun(self, intentions, states, q_vals):
        self.value_fun.train()

        intentions = intentions.to(self.device)
        states = states.to(self.device)
        q_vals = q_vals.to(self.device)

        for i in range(self.vf_iters):
            def mse():
                self.value_optimizer.zero_grad()
                state_values = self.value_fun(torch.cat((intentions, states), dim=-1)).view(-1)

                loss = self.mse_loss(state_values, q_vals)

                flat_params = get_flat_params(self.value_fun)
                l2_loss = self.vf_l2_reg_coef * torch.sum(torch.pow(flat_params, 2))
                loss += l2_loss

                loss.backward()

                return loss

            self.value_optimizer.step(mse)


    def update_policy(self, intentions, states, actions, advantages):
        self.policy.train()

        intentions = intentions.to(self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)

        action_dists = self.policy(torch.cat((intentions, states), dim=-1))
        log_action_probs = action_dists.log_prob(actions)

        loss = self.surrogate_loss(log_action_probs, log_action_probs.detach(), advantages)
        loss_grad = flat_grad(loss, self.policy.parameters(), retain_graph=True)

        mean_kl = mean_kl_first_fixed(action_dists, action_dists)

        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())
        search_dir = cg_solver(Fvp_fun, loss_grad, self.cg_max_iters)

        expected_improvement = torch.matmul(loss_grad, search_dir)

        def constraints_satisfied(step, beta):
            apply_update(self.policy, step)

            with torch.no_grad():
                new_action_dists = self.policy(torch.cat((intentions, states), dim=-1))
                new_log_action_probs = new_action_dists.log_prob(actions)

                new_loss = self.surrogate_loss(new_log_action_probs, log_action_probs, advantages)

                mean_kl = mean_kl_first_fixed(action_dists, new_action_dists)

            actual_improvement = new_loss - loss
            improvement_ratio = actual_improvement / (expected_improvement * beta)

            apply_update(self.policy, -step)

            surrogate_cond = improvement_ratio >= self.line_search_accept_ratio and actual_improvement > 0.0
            kl_cond = mean_kl <= self.max_kl_div

            return surrogate_cond and kl_cond

        max_step_len = self.get_max_step_len(search_dir, Fvp_fun, self.max_kl_div, retain_graph=True)
        step_len = line_search(search_dir, max_step_len, constraints_satisfied, self.device)

        opt_step = step_len * search_dir
        apply_update(self.policy, opt_step)

    def surrogate_loss(self, log_action_probs, imp_sample_probs, advantages):
        return torch.mean(torch.exp(log_action_probs - imp_sample_probs) * advantages)

    def get_max_step_len(self, search_dir, Hvp_fun, max_step, retain_graph=False):
        num = 2 * max_step
        denom = torch.matmul(search_dir, Hvp_fun(search_dir, retain_graph))
        max_step_len = torch.sqrt(num / denom)

        return max_step_len


