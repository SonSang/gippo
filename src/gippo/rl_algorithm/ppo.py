import torch as th
from typing import List

from gippo.rl_algorithm.base import RLAlgorithm
from gippo.dataset import PPODataset
from copy import deepcopy

class PPO(RLAlgorithm):

    def __init__(self, config, env_config, device="cpu", log_path=None):
        
        super(PPO, self).__init__(config, env_config, device, log_path)
        
        self.actor_lr = float(config["actor_learning_rate"])
        self.actor_optimizer = th.optim.Adam(
            self.actor.parameters(),
            lr = self.actor_lr,
            eps = 1e-8,
        )

        ppo_config = config.get("ppo", {})

        # clipping parameter for PPO updates;
        self.e_clip = float(ppo_config.get("e_clip", 0.2))
        
        # minibatch settings for PPO updates;
        self.mini_epochs = int(ppo_config.get("mini_epochs", 5))
        self.minibatch_size = int(ppo_config.get("minibatch_size", 
                                            self.horizon_length * self.num_actors))
        self.dataset = PPODataset(self.batch_size, 
                                self.minibatch_size, 
                                device)

        '''
        Measures to prevent false optimization.

        Theoretically, we optimize surrogate loss function for learning
        better policy. However, if learning rate is too large, the optimization
        result could be worse than the previous one. If such case is detected,
        we decrease the learning rate and try again. [max_optim_iter] denotes
        the maximum number of such cycles.
        '''
        # use backup actor to restore the previous policy when optimization fails;
        self.b_actor = deepcopy(self.actor)

        # maximum number of iterations for actor optimization;
        self.max_optim_iter = int(ppo_config.get("max_optim_iter", 8))

        # multiplier to decrease learning rate;
        self.learning_rate_multiplier = float(ppo_config.get("learning_rate_multiplier", 1.5))
        
    def train_actor_critic_no_ppo(self):

        return super().train_actor_critic_no_ppo()

    def use_analytic_grads(self):
        
        return False
    
    def use_ppo(self):

        return True

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

        pass

    def train_actor_ppo(self, batch_dict):

        self.prepare_dataset(batch_dict)

        # backup actor and optimizer to prevent policy degradation;
        self.backup_actor()

        initial_actor_lr = self.actor_lr

        for iter in range(self.max_optim_iter):
            
            a_losses = []
        
            for _ in range(0, self.mini_epochs):

                for i in range(len(self.dataset)):

                    a_loss, cmu, csigma = self.calc_gradients(self.dataset[i])
                    a_losses.append(a_loss)
                    self.dataset.update_mu_sigma(cmu, csigma)   

                    # this is erroneous code in original implementation,
                    # put here for fair reproducibility;
                    for param in self.actor_optimizer.param_groups:
                        param['lr'] = self.actor_lr

            first_mini_epoch_loss = th.stack(a_losses[:len(self.dataset)]).mean()
            last_mini_epoch_loss = th.stack(a_losses[-len(self.dataset):]).mean()

            if last_mini_epoch_loss > first_mini_epoch_loss:
                
                with th.no_grad():
                    
                    # optimization failed, restore the previous policy;
                    self.restore_actor()
                    
                    # decrease learning rate;
                    # @TODO: this is also an error in original implementation,
                    # put here for fair reproducibility;
                    for param in self.actor_optimizer.param_groups:
                        param['lr'] = initial_actor_lr / self.learning_rate_multiplier
                    self.actor_lr = initial_actor_lr / self.learning_rate_multiplier
            else:
                # @TODO: this is also an error in original implementation,
                # put here for fair reproducibility;
                self.actor_lr = initial_actor_lr
                break

        self.writer.add_scalar("info/actor_lr", self.actor_lr, self.epoch_num)

        return a_losses

    def prepare_dataset(self, batch_dict):

        obses = batch_dict['obses']
        advantages = batch_dict['advantages']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        
        advantages = th.sum(advantages, axis=1)
        
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['advantages'] = advantages
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        
        dataset_dict['old_mu'] = mus
        dataset_dict['old_sigma'] = sigmas
        dataset_dict['old_logp_actions'] = neglogpacs

        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['logp_actions'] = neglogpacs

        self.dataset.update_values_dict(dataset_dict)

    def backup_actor(self):

        with th.no_grad():
            for param, param_targ in zip(self.actor.parameters(), self.b_actor.parameters()):
                param_targ.data.mul_(0.)
                param_targ.data.add_(param.data)

    def restore_actor(self):

        with th.no_grad():
            for param, param_targ in zip(self.b_actor.parameters(), self.actor.parameters()):
                param_targ.data.mul_(0.)
                param_targ.data.add_(param.data)

    def calc_gradients(self, input_dict):

        advantage = input_dict['advantages']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        old_action_log_probs_batch = input_dict['old_logp_actions']         # original action log probs;
        curr_e_clip = self.e_clip

        # get current policy's actions;
        curr_mu, curr_std = self.actor.forward_dist(obs_batch)
        if curr_std.ndim == 1:
            curr_std = curr_std.unsqueeze(0)                      
            curr_std = curr_std.expand(curr_mu.shape[0], -1).clone()
        neglogp = self.neglogp(actions_batch, curr_mu, curr_std, th.log(curr_std))

        a_loss = self.actor_loss(old_action_log_probs_batch, 
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

    def actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = old_action_log_probs_batch - action_log_probs
        ratio = th.clamp(ratio, max=64.0)        # prevent ratio becoming [inf];
        ratio = th.exp(ratio)
        
        surr1 = advantage * ratio
        surr2 = advantage * th.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = th.max(-surr1, -surr2)
    
        return a_loss