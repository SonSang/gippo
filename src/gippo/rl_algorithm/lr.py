import torch as th
import torch.utils as tu
from typing import List

from gippo.rl_algorithm.base import RLAlgorithm
from gippo.utils import swap_and_flatten01

class LR(RLAlgorithm):

    def __init__(self, config, env_config, device="cpu", log_path=None):

        super(LR, self).__init__(config, env_config, device, log_path)
        
        self.actor_lr = float(config["actor_learning_rate"])
        self.actor_optimizer = th.optim.Adam(
            self.actor.parameters(), 
            betas = config['betas'], 
            lr = self.actor_lr
        )

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
        
        return False
    
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
        Train actor using Likelihood-Ratio (LR) techinque.

        There are two additional measures to reduce variance:
        1. Use advantage term instead of total expected return.
        (Using total expected return resulted in hopless results in some problems...)
        2. Normalize advantages (if [normalize_advantage] flag is set).
        '''

        self.actor.train()

        with th.no_grad():
            # compute advantages;
            curr_grad_advs = self.grad_advantages(self.tau, 
                                                grad_values, 
                                                grad_next_values,
                                                grad_rewards,
                                                grad_fdones,
                                                last_fdones)
        
            t_obses = swap_and_flatten01(th.stack(grad_obses, dim=0))
            t_advantages = swap_and_flatten01(th.stack(curr_grad_advs, dim=0))
            t_actions = swap_and_flatten01(th.stack(grad_actions, dim=0))
        
            # to reduce variance, we admit normalizing advantages;
            if self.normalize_advantage:
                t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)
            
        _, mu, std, _ = self.actor.forward_with_dist(t_obses)
        t_neglogpacs = self.neglogp(t_actions, mu, std, th.log(std))
                
        actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
        
        # divide by number of (s, a) pairs;
        actor_loss = th.mean(actor_loss)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.truncate_grads:
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
        self.actor_optimizer.step()