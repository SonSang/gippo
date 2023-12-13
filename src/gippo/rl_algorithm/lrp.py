import torch as th
import numpy as np
import torch.utils as tu
from typing import List

from gippo.rl_algorithm.base import RLAlgorithm
from gippo.utils import swap_and_flatten01

class LRP(RLAlgorithm):

    def __init__(self, config, env_config, device="cpu", log_path=None):
        
        super(LRP, self).__init__(config, env_config, device, log_path)
        
        self.actor_lr = float(config["actor_learning_rate"])
        self.actor_optimizer = th.optim.Adam(
            self.actor.parameters(), 
            betas = config['betas'], 
            lr = self.actor_lr
        )
        
        '''
        Parameters for sample variance estimation of policy gradients.
        '''
        # [var_est_num_sample]: Number of samples to use for sample variance estimation;
        self.var_est_num_sample = config.get("var_est_num_sample", 16)

        # [var_est_max_grad_len]: Length of the first N values (in policy gradients) to use
        # for sample variance estimation;
        self.var_est_max_grad_len = config.get("var_est_max_grad_len", 512)

    def train_actor_critic_no_ppo(self):

        '''
        Set learning rate.
        '''
        # set learning rate;
        actor_lr = self.actor_lr
        critic_lr = self.critic_lr
        if self.lr_schedule == 'linear':
            actor_lr = (1e-5 - self.actor_lr) * float(self.epoch_num / self.max_epochs) + self.actor_lr
            critic_lr = (1e-5 - self.critic_lr) * float(self.epoch_num / self.max_epochs) + self.critic_lr
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

        self.writer.add_scalar("info/actor_lr", actor_lr, self.epoch_num)
        self.writer.add_scalar("info/critic_lr", critic_lr, self.epoch_num)

        return super().train_actor_critic_no_ppo()

    def use_analytic_grads(self):
        
        return True
    
    def use_ppo(self):

        return False

    def get_optimizers_state(self):
        state = super().get_optimizers_state()
        state['actor'] = self.actor_optimizer.state_dict()
        
        return state

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
        '''
        Combine policy gradients obtained through LR and RP techinques.

        Use sample variance of the policy gradients to combine them.
        '''

        self.actor.train()

        lr_gradient_var = None
        rp_gradient_var = None

        '''
        Preliminaries
        '''
        # compute advantages;
        curr_grad_advs = self.grad_advantages(self.tau, 
                                            grad_values, 
                                            grad_next_values,
                                            grad_rewards,
                                            grad_fdones,
                                            last_fdones)
        
        '''
        Estimate LR gradients and their variances.
        '''

        with th.no_grad():
            t_obses = swap_and_flatten01(th.stack(grad_obses, dim=0))
            t_advantages = swap_and_flatten01(th.stack(curr_grad_advs, dim=0))
            t_actions = swap_and_flatten01(th.stack(grad_actions, dim=0))
            
            # to reduce variance, we admit normalizing advantages;
            if self.normalize_advantage:
                t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

        _, mu, std, _ = self.actor.forward_with_dist(t_obses)
        t_neglogpacs = self.neglogp(t_actions, mu, std, th.log(std))
            
        actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
        
        # randomly select subset to compute sample variance;
        sample_num = np.min([self.var_est_num_sample, len(actor_loss)])  #if len(actor_loss) > 64 else len(actor_loss)
        actor_loss_num = len(actor_loss)
        actor_loss_indices = th.randperm(actor_loss_num)[:sample_num]
        lr_gradients = []
        for ai in actor_loss_indices:
            al = actor_loss[ai].sum()
            
            self.actor_optimizer.zero_grad()
            al.backward(retain_graph=True)
            assert len(self.actor_optimizer.param_groups) == 1, ""
            grad_list = []
            for param in self.actor_optimizer.param_groups[0]['params']:
                grad_list.append(param.grad.reshape([-1]))
            grad = th.cat(grad_list)
            
            # if length of the gradient is too long, we truncate it
            # because it is too time consuming to use all of the gradients;
            if len(grad) > self.var_est_max_grad_len:
                grad = grad[:self.var_est_max_grad_len]
            lr_gradients.append(grad)
            
        lr_gradients = th.stack(lr_gradients, dim=0)
            
        lr_gradient_cov = th.cov(lr_gradients.transpose(0, 1))
        if lr_gradient_cov.ndim == 0:
            lr_gradient_cov = lr_gradient_cov.unsqueeze(0).unsqueeze(0)
        lr_gradient_var = lr_gradient_cov.diagonal(0).sum()

        '''
        Estimate RP gradients and their variances.
        '''    
        
        # add value of the states;
        for i in range(len(grad_values)):
            curr_grad_advs[i] = curr_grad_advs[i] + grad_values[i]

        rp_gradients = []
        for i in range(grad_start.shape[0]):
            for j in range(grad_start.shape[1]):
                if not grad_start[i, j]:
                    continue
                
                al: th.Tensor = -curr_grad_advs[i][j].sum()
                
                self.actor_optimizer.zero_grad()
                al.backward(retain_graph=True)
                assert len(self.actor_optimizer.param_groups) == 1, ""
                grad_list = []
                for param in self.actor_optimizer.param_groups[0]['params']:
                    grad_list.append(param.grad.reshape([-1]))
                grad = th.cat(grad_list)

                # if length of the gradient is too long, we truncate it
                # because it is too time consuming to use all of the gradients;
                if len(grad) > self.var_est_max_grad_len:
                    grad = grad[:self.var_est_max_grad_len]
                rp_gradients.append(grad)
                
                if len(rp_gradients) >= self.var_est_num_sample:
                    break
            
            if len(rp_gradients) >= self.var_est_num_sample:
                break
                
        rp_gradients = th.stack(rp_gradients, dim=0)
            
        rp_gradient_cov = th.cov(rp_gradients.transpose(0, 1))
        if rp_gradient_cov.ndim == 0:
            rp_gradient_cov = rp_gradient_cov.unsqueeze(0).unsqueeze(0)
        rp_gradient_var = rp_gradient_cov.diagonal(0).sum()
            
        '''
        Interpolate LR and RP gradients using sample variances.
        '''
        k_lr = (rp_gradient_var) / (lr_gradient_var + rp_gradient_var + 1e-8)
        k_rp = 1. - k_lr
        
        # self.writer.add_scalar("info/basic_k_lr", k_lr, self.epoch_num)
        
        lr_actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
        lr_actor_loss = th.mean(lr_actor_loss)
        
        rp_actor_loss = -self.grad_advantages_first_terms_sum(curr_grad_advs, grad_start)
        rp_actor_loss = rp_actor_loss / th.count_nonzero(grad_start)
        
        actor_loss = (lr_actor_loss * k_lr) + (rp_actor_loss * k_rp)
        
        # update actor;
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.truncate_grads:
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
        self.actor_optimizer.step()