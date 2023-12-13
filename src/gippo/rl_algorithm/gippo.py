import torch as th
import numpy as np
from typing import List

from gippo.rl_algorithm.ppo import PPO
from gippo.utils import swap_and_flatten01, Normal, RunningMeanStd

class GIPPO(PPO):

    def __init__(self, config, env_config, device="cpu", log_path=None):
        
        super(GIPPO, self).__init__(config, env_config, device, log_path)
        
        '''
        Use different optimizers for analytical gradient-based 
        actor update and PPO-based actor update
        @TODO: Merge two optimizers?
        '''
        self.actor_lr_no_ppo = float(config["actor_learning_rate_no_ppo"])
        self.actor_optimizer_no_ppo = th.optim.Adam(
            self.actor.parameters(), 
            betas = config['betas'], 
            lr = self.actor_lr_no_ppo
        )

        '''
        Parameters for alpha-policy updates.
        '''
        gi_config = config.get("gi", {})
        self.gi_alpha = float(gi_config.get("alpha", 1e-3))
        self.gi_alpha_interval = float(gi_config.get("alpha_interval", 0.2))
        self.gi_alpha_update_factor = float(gi_config.get("alpha_update_factor", 1.1))
        self.gi_max_alpha = float(gi_config.get("max_alpha", 1.0))
        self.gi_num_iter = int(gi_config.get("num_iter", 16))
        self.gi_max_oorr = float(gi_config.get("max_oorr", 0.5))
        
        # rms for estimated alpha-policy performance;
        self.est_alpha_performace_rms = RunningMeanStd()
        
    def use_analytic_grads(self):
        
        return True
    
    def use_ppo(self):

        return True

    def get_optimizers_state(self):
        state = super().get_optimizers_state()
        state['actor_no_ppo'] = self.actor_optimizer_no_ppo.state_dict()
        
        return state

    def train_actor_critic_no_ppo(self):

        '''
        Set learning rate.
        '''
        # set learning rate;
        # do not change actor learning rate;
        # @TODO: too messy code, and errorneous;
        actor_lr_no_ppo = self.actor_lr_no_ppo
        critic_lr = self.critic_lr
        if self.lr_schedule == 'linear':
            critic_lr = (1e-5 - self.critic_lr) * float(self.epoch_num / self.max_epochs) + self.critic_lr
        
        for param_group in self.actor_optimizer_no_ppo.param_groups:
            param_group['lr'] = actor_lr_no_ppo
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

        self.writer.add_scalar("info/actor_lr_no_ppo", self.actor_lr_no_ppo, self.epoch_num)
        self.writer.add_scalar("info/critic_lr", critic_lr, self.epoch_num)
        self.writer.add_scalar("gi_info/alpha", self.gi_alpha, self.epoch_num)

        return super().train_actor_critic_no_ppo()

    def train_actor_no_ppo(self,
                            grad_start: th.Tensor,
                            grad_obses: List[th.Tensor],
                            grad_rp_eps: List[th.Tensor],
                            grad_actions: List[th.Tensor],
                            grad_values: List[th.Tensor],
                            grad_next_values: List[th.Tensor],
                            grad_rewards: List[th.Tensor],
                            grad_fdones: List[th.Tensor],
                            last_fdones: th.Tensor):

        # compute advantages;
        curr_grad_advs = self.grad_advantages(self.tau, 
                                            grad_values, 
                                            grad_next_values,
                                            grad_rewards,
                                            grad_fdones,
                                            last_fdones)
            
        # compute gradients of advantages w.r.t. actions;
        t_adv_gradient = self.differentiate_grad_advantages(grad_actions,
                                                            curr_grad_advs,
                                                            grad_start,
                                                            False)

        with th.no_grad():
        
            t_obses = swap_and_flatten01(th.stack(grad_obses, dim=0))
            t_rp_eps = swap_and_flatten01(th.stack(grad_rp_eps, dim=0))
            
            t_advantages = swap_and_flatten01(th.stack(curr_grad_advs, dim=0))
            t_actions = swap_and_flatten01(th.stack(grad_actions, dim=0))
            t_adv_gradient = swap_and_flatten01(t_adv_gradient)
            t_alpha_actions = t_actions + self.gi_alpha * t_adv_gradient
            
            # write log about variance;
            # advantage variance;
            t_advantages_var = th.var(t_advantages, dim=0)
            t_adv_gradient_cov = th.cov(t_adv_gradient.transpose(0, 1))
            if t_adv_gradient_cov.ndim == 0:
                t_adv_gradient_cov = t_adv_gradient_cov.unsqueeze(0).unsqueeze(0)
            t_adv_gradient_var = t_adv_gradient_cov.diagonal(0).sum()
            
            self.writer.add_scalar("gi_info/advantage_variance", t_advantages_var, self.epoch_num)
            self.writer.add_scalar("gi_info/advantage_gradient_variance", t_adv_gradient_var, self.epoch_num)
                
        # backup actor before actor update;
        self.backup_actor()

        '''
        Update policy to alpha-policy.
        '''
        for i in range(self.max_optim_iter):
        
            actor_loss_0 = None
            actor_loss_1 = None

            for j in range(self.gi_num_iter):
                
                _, mu, std, _ = self.actor.forward_with_dist(t_obses)
                
                distr = Normal(mu, std)
                rpeps_actions = distr.eps_to_action(t_rp_eps)
                
                actor_loss = (rpeps_actions - t_alpha_actions) * (rpeps_actions - t_alpha_actions)
                actor_loss = th.sum(actor_loss, dim=-1)
                actor_loss = actor_loss.mean()
                
                # update actor;
                self.actor_optimizer_no_ppo.zero_grad()
                actor_loss.backward()
                if self.truncate_grads:
                    th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
                self.actor_optimizer_no_ppo.step()
                
                if j == 0:
                    actor_loss_0 = actor_loss.detach().cpu().item()
                elif j == self.gi_num_iter - 1:
                    actor_loss_1 = actor_loss.detach().cpu().item()
                    
            log_actor_loss_0 = np.log(actor_loss_0)
            log_actor_loss_1 = np.log(actor_loss_1)
            actor_loss_ratio = np.exp(log_actor_loss_1 - log_actor_loss_0)
            
            if actor_loss_0 < actor_loss_1:
                
                with th.no_grad():
                    
                    # if optimization did not work well, restore original
                    # policy, decrease learning rate and try again;
                    self.restore_actor()

                    for param in self.actor_optimizer_no_ppo.param_groups:
                        param['lr'] /= self.learning_rate_multiplier

                    continue

            else:

                # @TODO: errorneous code, put here for fair reproducibility;
                for param in self.actor_optimizer_no_ppo.param_groups:
                    param['lr'] = self.actor_lr_no_ppo

                break
        
        self.writer.add_scalar("gi_info/actor_loss_ratio", actor_loss_ratio, self.epoch_num)
                
        did_converge = actor_loss_0 > actor_loss_1
                
        '''
        Estimate determinant of (I + alpha * advantage Hessian)
        and use it to safely bound alpha.
        '''
        with th.no_grad():

            old_mu, old_std = self.experience_buffer.tensor_dict['mus'], \
                                self.experience_buffer.tensor_dict['sigmas']
                                
            old_mu, old_std = swap_and_flatten01(old_mu), swap_and_flatten01(old_std)
                                
            _, new_mu, new_std, _ = self.actor.forward_with_dist(t_obses)
            
        preupdate_action_eps_jac = self.action_eps_jacobian(old_mu, old_std, t_rp_eps)
        postupdate_action_eps_jac = self.action_eps_jacobian(new_mu, new_std, t_rp_eps)
        
        preupdate_action_eps_jacdet = th.logdet(preupdate_action_eps_jac)
        postupdate_action_eps_jacdet = th.logdet(postupdate_action_eps_jac)
        
        est_hessian_logdet = postupdate_action_eps_jacdet - preupdate_action_eps_jacdet
        est_hessian_det = th.exp(est_hessian_logdet)
        
        mean_est_hessian_det = th.mean(est_hessian_det)
        min_est_hessian_det = th.min(est_hessian_det)
        max_est_hessian_det = th.max(est_hessian_det)
        
        self.writer.add_scalar("gi_info/mean_est_hessian_det", mean_est_hessian_det, self.epoch_num)
        self.writer.add_scalar("gi_info/min_est_hessian_det", min_est_hessian_det, self.epoch_num)
        self.writer.add_scalar("gi_info/max_est_hessian_det", max_est_hessian_det, self.epoch_num)
        
        '''
        Update alpha and actor learning rate for next iteration.
        '''
        curr_alpha = self.gi_alpha
        curr_actor_lr_no_ppo = self.actor_lr_no_ppo
        
        next_alpha = curr_alpha
        next_actor_lr_no_ppo = curr_actor_lr_no_ppo
        
        # we have to keep [est_hessian_det] in this range;
        min_safe_interval = (1. - self.gi_alpha_interval)
        max_safe_interval = (1. + self.gi_alpha_interval)
        
        if not did_converge:
            # alpha does not change, only decrease actor learning rate;
            next_actor_lr_no_ppo = curr_actor_lr_no_ppo / self.learning_rate_multiplier
        else:
            # actor_lr does not change, only change alpha;
            if min_est_hessian_det < min_safe_interval or \
                max_est_hessian_det > max_safe_interval:
                next_alpha = curr_alpha / self.gi_alpha_update_factor
            else:
                next_alpha = curr_alpha * self.gi_alpha_update_factor
                
        next_alpha = np.clip(next_alpha, None, self.gi_max_alpha)
        next_actor_lr_no_ppo = np.clip(next_actor_lr_no_ppo, 1e-5, None)

        '''
        Observe how much alpha-policy is different from the original
        policy, and then adjust [next_alpha] accordingly.
        '''
        next_alpha = self.adjust_next_alpha_by_policy_diff(next_alpha, curr_grad_advs)

        self.gi_alpha = next_alpha
        self.actor_lr_no_ppo = next_actor_lr_no_ppo

        return

    def differentiate_grad_advantages(self, 
                                    grad_actions: th.Tensor, 
                                    grad_advs: th.Tensor, 
                                    grad_start: th.Tensor, 
                                    debug=False):
        
        '''
        Compute first-order gradients of [grad_advs] w.r.t. 
        [grad_actions] using automatic differentiation.
        '''

        num_timestep = grad_start.shape[0]
        num_actor = grad_start.shape[1]

        '''
        Using GAE, we can compute gradient of [grad_advs] at each
        time step by only backpropagating once for the first time
        step of a trajectory.
        '''
        adv_sum: th.Tensor = self.grad_advantages_first_terms_sum(grad_advs, grad_start)
        for ga in grad_actions:
            ga.retain_grad()
        adv_sum.backward(retain_graph=debug)
        adv_gradient = []
        for ga in grad_actions:
            adv_gradient.append(ga.grad)
        adv_gradient = th.stack(adv_gradient)
        
        # reweight gradients, so that we get correct gradients
        # for each time step;
        with th.no_grad():

            c = (1.0 / (self.gamma * self.tau))
            cv = th.ones((num_actor, 1), device=adv_gradient.device)

            for nt in range(num_timestep):

                # if new episode has been started, set [cv] to 1; 
                for na in range(num_actor):
                    if grad_start[nt, na]:
                        cv[na, 0] = 1.0

                adv_gradient[nt] = adv_gradient[nt] * cv
                cv = cv * c
                
        if debug:

            '''
            Compute gradients of [grad_advs] at each time step
            in brute force and compare it with the above computation
            results, which is more efficient than this.
            '''
            for i in range(num_timestep):
                
                debug_adv_sum = grad_advs[i].sum()
                
                debug_grad_adv_gradient = th.autograd.grad(debug_adv_sum, grad_actions[i], retain_graph=True)[0]
                debug_grad_adv_gradient_norm = th.norm(debug_grad_adv_gradient, p=2, dim=-1)
                
                debug_grad_error = th.norm(debug_grad_adv_gradient - adv_gradient[i], p=2, dim=-1)
                debug_grad_error_ratio = debug_grad_error / debug_grad_adv_gradient_norm
                
                assert th.all(debug_grad_error_ratio < 0.01), \
                    "Gradient of advantage possibly wrong"
                        
        adv_gradient = adv_gradient.detach()
        
        return adv_gradient

    def action_eps_jacobian(self, mu, sigma, eps):

        '''
        Assume action is computed as:
        a = mu + sigma * eps,
        where mu, sigma, and eps are all one-dim tensors.
        '''
        
        jacobian = th.zeros((eps.shape[0], eps.shape[1], eps.shape[1]))
        
        for d in range(eps.shape[1]):
            
            if sigma.ndim == 1:
                jacobian[:, d, d] = sigma[d].detach()
            elif sigma.ndim == 2:
                jacobian[:, d, d] = sigma[:, d].detach()
            
        return jacobian

    @th.no_grad()
    def adjust_next_alpha_by_policy_diff(self, next_alpha, grad_advs):
        '''
        Observe how much alpha-policy is different from the original
        policy, and then adjust [next_alpha] accordingly.

        If alpha-policy is too far away from the original policy,
        decrease [next_alpha], so that there is some room for PPO
        optimization.
        '''
        obses = swap_and_flatten01(self.experience_buffer.tensor_dict['obses'].detach())
        neglogpacs = swap_and_flatten01(self.experience_buffer.tensor_dict['neglogpacs'].detach())
        actions = swap_and_flatten01(self.experience_buffer.tensor_dict['actions'].detach())
        advantages = swap_and_flatten01(th.cat(grad_advs, dim=0).detach())

        n_mus, n_sigmas = self.actor.forward_dist(obses)
        if n_sigmas.ndim == 1:
            n_sigmas = n_sigmas.unsqueeze(0)                      
            n_sigmas = n_sigmas.expand(n_mus.shape[0], -1).clone()
        
        n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, th.log(n_sigmas))
        
        '''
        Estimate difference between alpha-policy and original policy
        using out-of-range-ratio.
        '''
        pac_ratio = th.exp(th.clamp(neglogpacs - n_neglogpacs, max=16.))  # prevent [inf];
        out_of_range_pac_ratio = th.logical_or(pac_ratio < (1. - self.e_clip), 
                                                    pac_ratio > (1. + self.e_clip))
        out_of_range_pac_ratio = th.count_nonzero(out_of_range_pac_ratio) / actions.shape[0]
        
        self.writer.add_scalar("gi_info/out_of_range_ratio", out_of_range_pac_ratio, self.epoch_num)
                
        '''
        Evaluate the bias of analytical gradients by estimating the
        performance of alpha-policy in terms of PPO.
        '''
        est_alpha_performance = \
            th.sum(advantages * pac_ratio) - \
            th.sum(advantages)
        
        # @TODO: Ugly approach to prevent overly noisy [est_alpha_performance];
        n_est_alpha_performance = self.est_alpha_performace_rms.normalize(est_alpha_performance)
        self.est_alpha_performace_rms.update(est_alpha_performance.unsqueeze(0))
        
        self.writer.add_scalar("gi_info/est_alpha_performance", est_alpha_performance, self.epoch_num)
        self.writer.add_scalar("gi_info/est_alpha_performance_normalized", n_est_alpha_performance, self.epoch_num)
                
        '''
        In following conditions, decrease [next_alpha]:
        1. [out_of_range_pac_ratio] is too high (guarantee PPO update);
        2. [est_alpha_performance] is negative (biased analytical grads);
        '''
        if out_of_range_pac_ratio > self.gi_max_oorr or \
            (est_alpha_performance < 0 and n_est_alpha_performance < -1.):
            
            next_alpha = self.gi_alpha / self.gi_alpha_update_factor
        
        next_alpha = np.clip(next_alpha, None, self.gi_max_alpha)
        return next_alpha

    def prepare_dataset(self, batch_dict):

        super().prepare_dataset(batch_dict)

        '''
        Since policy could have been updated to alpha policy,
        change [mu], [sigma], and [logp_actions] accordingly.
        '''
        obses = batch_dict['obses']
        actions = batch_dict['actions']

        with th.no_grad():
            n_mus, n_sigmas = self.actor.forward_dist(obses)
            if n_sigmas.ndim == 1:
                n_sigmas = n_sigmas.unsqueeze(0)                      
                n_sigmas = n_sigmas.expand(n_mus.shape[0], -1).clone()    
            n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, th.log(n_sigmas))
        
        self.dataset.values_dict['mu'] = n_mus
        self.dataset.values_dict['sigma'] = n_sigmas
        self.dataset.values_dict['logp_actions'] = n_neglogpacs

    def calc_gradients(self, input_dict):

        advantage = input_dict['advantages']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        old_action_log_probs_batch_before_alpha = input_dict['old_logp_actions']      # action log probs before alpha update;
        old_action_log_probs_batch_after_alpha = input_dict['logp_actions']           # action log probs after alpha update;
        
        curr_e_clip = self.e_clip

        # get current policy's actions;
        curr_mu, curr_std = self.actor.forward_dist(obs_batch)
        if curr_std.ndim == 1:
            curr_std = curr_std.unsqueeze(0)                      
            curr_std = curr_std.expand(curr_mu.shape[0], -1).clone()
        neglogp = self.neglogp(actions_batch, curr_mu, curr_std, th.log(curr_std))

        a_loss = self.actor_loss(old_action_log_probs_batch_before_alpha,
                                old_action_log_probs_batch_after_alpha, 
                                neglogp, 
                                advantage,
                                curr_e_clip).mean()

        # we only use actor loss here for fair comparison;
        loss = a_loss
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.truncate_grads:
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()
   
        self.train_result = (a_loss, curr_mu.detach(), curr_std.detach())

        return self.train_result

    def actor_loss(self, 
                old_action_log_probs_batch_before_alpha, 
                old_action_log_probs_batch_after_alpha,
                action_log_probs, 
                advantage, 
                curr_e_clip):
        
        t_ratio = old_action_log_probs_batch_before_alpha - \
                    old_action_log_probs_batch_after_alpha
        
        if th.any(th.abs(t_ratio) > 4.):
            # ratio can be numerically unstable, just use original ppo;
            # but use policy after RP update as importance sampling distribution;
            ratio = old_action_log_probs_batch_after_alpha - action_log_probs
        else:
            t_ratio = th.exp(t_ratio)
            tmp0 = th.log(t_ratio + 1.)
            tmp1 = tmp0 - old_action_log_probs_batch_before_alpha
            action_log_probs_batch_mid = np.log(2.) - tmp1
            
            ratio = action_log_probs_batch_mid - action_log_probs
            
        ratio = th.clamp(ratio, min=-16., max=16.)        # prevent ratio becoming [inf];
        ratio = th.exp(ratio)

        surr1 = advantage * ratio
        surr2 = advantage * th.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = th.max(-surr1, -surr2)
        
        return a_loss