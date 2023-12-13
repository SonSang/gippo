'''
Modified from 
https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/a2c_continuous.py
'''

import numpy as np
import torch as th
from torch import nn
from typing import List

import time
import gym
import copy
import os

from gippo.utils import RunningMeanStd, AverageMeter, swap_and_flatten01
from gippo.vecenv import create_vecenv
from gippo.network import ActorStochasticMLP, CriticMLP
from gippo.experience import ExperienceBuffer
from gippo.dataset import CriticDataset

from torch.utils.tensorboard import SummaryWriter

save_distribution = False

class RLAlgorithm:
    def __init__(self, 
                config, 
                env_config,
                device="cpu", 
                log_path=None):

        '''
        Basic configs
        '''
        self.config = config
        self.device = device

        # logging;
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = f'./logdir/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        self.nn_dir = os.path.join(self.log_path, 'nn')
        self.summaries_dir = os.path.join(self.log_path, 'runs')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)

        # experience buffer size that we are going to use for training;
        self.horizon_length = config.get('horizon_length', 32)
        self.num_actors = config.get('num_actors', 1)
        self.num_agents = config.get('num_agents', 1)
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors

        # env configs;
        self.env_name = env_config['name']
        self.env_config = env_config.get('config', {})
        self.env_config['device'] = self.device
        self.env_config['no_grad'] = not self.use_analytic_grads()
        self.vec_env = create_vecenv(
            self.env_name, 
            self.num_actors, 
            **self.env_config)
        self.env_info = self.vec_env.get_env_info()

        self.value_size = self.env_info.get('value_size', 1)

        # reshaper and normalization;
        self.rewards_shaper = config.get("rewards_shaper", None)
        self.normalize_input = config.get("normalize_input", False)
        self.normalize_value = config.get("normalize_value", False)
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.normalize_advantage = config.get("normalize_advantage", False)

        # observation;
        self.observation_space = self.env_info['observation_space']
        self.obs_shape = self.observation_space.shape
        self.obs = None

        # running stats;
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0

        # training;
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.network = config.get("network", None)

        '''
        Our work solves stochastic optimization problem in differentiable environment.
        '''
        num_obs = self.obs_shape[0]
        num_actions = self.env_info['action_space'].shape[0]

        self.actor = ActorStochasticMLP(num_obs, 
                                        num_actions, 
                                        config['network'], 
                                        device=self.device)

        self.critic = CriticMLP(num_obs,
                                config['network'],
                                device=self.device)

        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic_alpha = config.get('target_critic_alpha', 0.4)
        
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())

        '''
        Optimizers
        '''
            
        # critic;
        self.critic_lr = float(config["critic_learning_rate"])
        self.critic_optimizer = th.optim.Adam(
            self.critic.parameters(), 
            betas = config['betas'], 
            lr = self.critic_lr
        )
        self.critic_iterations = config["critic_iterations"]
        self.critic_num_batch = config["critic_num_batch"]

        # misc;
        self.truncate_grads = config["truncate_grads"]
        self.grad_norm = config["grad_norm"]

        # learning rate scheduler;
        self.lr_schedule = config['lr_schedule']
        
        # change to proper running mean std for backpropagation;
        if self.normalize_input:
            if isinstance(self.observation_space, gym.spaces.Dict):
                raise NotImplementedError()
            else:
                self.obs_rms = RunningMeanStd(shape=self.obs_shape, device=self.device)
                
        if self.normalize_value:
            self.val_rms = RunningMeanStd(shape=(1,), device=self.device)

        # episode length;
        self.episode_max_length = self.vec_env.env.episode_length

        # statistics;
        self.games_to_track = 100
        self.game_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)

        # GAE params;
        self.gamma = config['gamma']
        self.tau = config['tau']
        
    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        while True:
            epoch_num = self.update_epoch()
            
            step_time, no_ppo_time, ppo_time, sum_time, ppo_loss = \
                self.train_epoch()
            
            total_time += sum_time
            frame = self.frame

            # cleaning memory to optimize space
            if self.use_ppo():
                self.dataset.update_values_dict(None)
            
            print(f"Num steps: {frame + self.curr_frames}")

            # do we need scaled_time?
            scaled_time = sum_time #self.num_agents * sum_time
            scaled_no_ppo_time = no_ppo_time #self.num_agents * play_time
            curr_frames = self.curr_frames
            self.frame += curr_frames

            self.write_stats(total_time, 
                            epoch_num, 
                            step_time, 
                            no_ppo_time, 
                            ppo_time, 
                            ppo_loss,
                            frame, 
                            scaled_time, 
                            scaled_no_ppo_time, 
                            curr_frames)
            
            mean_rewards = [0]
            mean_lengths = 0

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()
                self.mean_rewards = mean_rewards[0]

                for i in range(self.value_size):
                    rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                    self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                    self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                    self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                checkpoint_name = self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)

                if self.save_freq > 0:
                    if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards[0]
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    
            if epoch_num > self.max_epochs:
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                print('MAX EPOCHS NUM!')
                return self.last_mean_rewards, epoch_num

            update_time = 0
            if self.print_stats:
                fps_step = curr_frames / step_time
                # fps_step_inference = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                # print(f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f} mean reward: {mean_rewards[0]:.2f} mean lengths: {mean_lengths:.1f}')
                print(f'epoch: {epoch_num} fps step: {fps_step:.1f} fps total: {fps_total:.1f} mean reward: {mean_rewards[0]:.2f} mean lengths: {mean_lengths:.1f}')

    def init_tensors(self):

        # use specialized experience buffer;
        batch_size = self.num_agents * self.num_actors
        
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
        }

        self.experience_buffer = ExperienceBuffer(
            self.env_info, 
            algo_info, 
            self.device
        )

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = th.zeros(current_rewards_shape, dtype=th.float32, device=self.device)
        self.current_lengths = th.zeros(batch_size, dtype=th.float32, device=self.device)
        self.dones = th.ones((batch_size,), dtype=th.uint8, device=self.device)

        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones', 'adv_grads']

    def cast_obs(self, obs):
        if isinstance(obs, th.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = th.ByteTensor(obs).to(self.device)
            else:
                obs = th.FloatTensor(obs).to(self.device)
        return obs
    
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            raise NotImplementedError()
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs
     
    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
    
    def train_epoch(self):

        self.vec_env.set_train_info(self.frame, self)

        no_ppo_time_start = time.time()

        # set learning rate;
        # if self.gi_lr_schedule == 'linear':
        #     if self.gi_algorithm in ['shac-only', 'grad-ppo-shac', 'basic-lr', 'basic-rp', 'basic-combination']:
        #         actor_lr = (1e-5 - self.actor_lr) * float(self.epoch_num / self.max_epochs) + self.actor_lr
        #     else:
        #         actor_lr = self.actor_lr
        #     critic_lr = (1e-5 - self.critic_lr) * float(self.epoch_num / self.max_epochs) + self.critic_lr
        # else:
        #     actor_lr = self.actor_lr
        #     critic_lr = self.critic_lr
        
        # for param_group in self.actor_optimizer.param_groups:
        #     param_group['lr'] = actor_lr
        # for param_group in self.critic_optimizer.param_groups:
        #     param_group['lr'] = critic_lr

        # self.writer.add_scalar("info/gi_actor_lr", actor_lr, self.epoch_num)
        # self.writer.add_scalar("info/gi_critic_lr", critic_lr, self.epoch_num)
        
        # # rp actor lr and alpha;
        
        # self.writer.add_scalar("info_alpha/actor_lr", self.actor_lr, self.epoch_num)
        # self.writer.add_scalar("info_alpha/alpha", self.gi_curr_alpha, self.epoch_num)

        '''
        Train actor critic using methods other than PPO.
        When we use PPO-based methods (PPO, GI-PPO),
        we additionally collect experience to use in PPO
        updates afterwards.
        '''
        batch_dict = self.train_actor_critic_no_ppo()
        no_ppo_time_end = time.time()

        self.curr_frames = batch_dict.pop('played_frames')
        
        '''
        Train actor using PPO-based algorithms using 
        collected experience above.
        '''
        ppo_time_start = time.time()
        if self.use_ppo():
            ppo_loss = self.train_actor_ppo(batch_dict)
        else:
            # placeholders;
            ppo_loss = [th.zeros((1,), dtype=th.float32, device=self.device)]
        ppo_time_end = time.time()

        no_ppo_time = no_ppo_time_end - no_ppo_time_start
        ppo_time = ppo_time_end - ppo_time_start
        total_time = ppo_time_end - no_ppo_time_start
        
        # update (rp) alpha and actor lr;
        
        # self.gi_curr_alpha = self.next_alpha
        # self.actor_lr = self.next_actor_lr

        return batch_dict['step_time'], \
                no_ppo_time, \
                ppo_time, \
                total_time, \
                ppo_loss

    def train_actor_critic_no_ppo(self):

        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        # indicator for steps that grad computation starts;
        grad_start = th.zeros_like(self.experience_buffer.tensor_dict['dones'])
        
        grad_obses = []
        grad_values = []
        grad_next_values = []
        grad_actions = []
        grad_rewards = []
        grad_fdones = []
        grad_rp_eps = []

        # use frozen [obs_rms] and [value_rms] during this one function call;
        curr_obs_rms = None
        curr_val_rms = None
        if self.normalize_input:
            with th.no_grad():
                curr_obs_rms = copy.deepcopy(self.obs_rms)
        if self.normalize_value:
            with th.no_grad():
                curr_val_rms = copy.deepcopy(self.val_rms)

        # start with clean grads;
        self.obs = self.vec_env.env.initialize_trajectory()
        self.obs = self.obs_to_tensors(self.obs)
        grad_start[0, :] = 1.0

        for n in range(self.horizon_length):

            if n > 0:
                grad_start[n, :] = self.dones

            # get action for current observation;
            if self.use_analytic_grads():
                res_dict = self.get_action_values(
                    self.obs, 
                    curr_obs_rms, 
                    curr_val_rms
                )
            else:
                with th.no_grad():
                    res_dict = self.get_action_values(
                        self.obs, 
                        curr_obs_rms, 
                        curr_val_rms
                    )
                
            # we store tensor objects with gradients;
            grad_obses.append(res_dict['obs'])
            grad_values.append(res_dict['values'])
            grad_actions.append(res_dict['actions'])
            grad_fdones.append(self.dones.float())
            grad_rp_eps.append(res_dict['rp_eps'])

            # [obs] is an observation of the current time step;
            # store processed obs, which might have been normalized already;
            self.experience_buffer.update_data('obses', n, res_dict['obs'])

            # [dones] indicate if this step is the start of a new episode;
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            
            # take action;
            step_time_start = time.time()
            actions = th.tanh(grad_actions[-1])
            
            if self.use_analytic_grads():
                self.obs, rewards, self.dones, infos = self.vec_env.step(actions)
            else:
                with th.no_grad():
                    self.obs, rewards, self.dones, infos = self.vec_env.step(actions)
            
            self.obs = self.obs_to_tensors(self.obs)
            rewards = rewards.unsqueeze(-1)
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            # compute value of next state;
            if self.use_analytic_grads():
                next_obs = infos['obs_before_reset']
            else:
                next_obs = self.obs['obs']
            
            if self.normalize_input:
                # do not update rms here;
                next_obs = curr_obs_rms.normalize(next_obs)
            next_value = self.target_critic(next_obs)
            if self.normalize_value:
                next_value = curr_val_rms.normalize(next_value, True)
            
            # even though [next_value] can wrong when it is based on
            # a [next_obs] that is at the start of new episode,
            # we deal with it by making it zero when it was an early termination;
            grad_next_values.append(next_value)

            done_env_ids = self.dones.nonzero(as_tuple = False).squeeze(-1)
            for id in done_env_ids:
                if th.isnan(next_obs[id]).sum() > 0 \
                    or th.isinf(next_obs[id]).sum() > 0 \
                    or (th.abs(next_obs[id]) > 1e6).sum() > 0: # ugly fix for nan values
                    grad_next_values[-1][id] = 0.
                elif self.current_lengths[id] < self.episode_max_length - 1:    # early termination
                    grad_next_values[-1][id] = 0.
            
            # add default reward;
            grad_rewards.append(rewards)

            # @TODO: do not use reward shaper for now;
            self.experience_buffer.update_data('rewards', n, rewards)

            self.current_rewards += rewards.detach()
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        '''
        Update actor and critic networks (but no PPO yet).

        Actor update differs between different algorithms,
        but critic update is shared between all algorithms.
        '''

        # start and end of current subsequence;
        last_fdones = self.dones.float()

        self.train_actor_no_ppo(grad_start,
                                grad_obses,
                                grad_rp_eps,
                                grad_actions,
                                grad_values,
                                grad_next_values,
                                grad_rewards,
                                grad_fdones,
                                last_fdones)
        
        grad_advs = \
            self.train_critic(grad_obses,
                            grad_actions,
                            grad_values,
                            grad_next_values,
                            grad_rewards,
                            grad_fdones,
                            last_fdones)

        self.update_target_critic()
        self.clear_experience_buffer_grads()

        # sort out [batch_dict];
        with th.no_grad():

            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

            for i in range(len(grad_advs)):
                grad_advs[i] = grad_advs[i].unsqueeze(0)
            batch_dict['advantages'] = swap_and_flatten01(th.cat(grad_advs, dim=0).detach())
            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time

        return batch_dict

    def use_analytic_grads(self):
        '''
        Whether current RL algorithm requires analytic gradients
        from differentiable environment.
        '''
        raise NotImplementedError()

    def neglogp(self, x, mean, std, logstd):
        '''
        Negative log probability of a batch of actions under a Gaussian policy.
        '''

        assert x.ndim == 2 and mean.ndim == 2 and std.ndim == 2 and logstd.ndim == 2, ""
        # assert x.shape[0] == mean.shape[0] and x.shape[0] == std.shape[0] and x.shape[0] == logstd.shape[0], ""

        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)

    def get_action_values(self, 
                        obs, 
                        obs_rms: RunningMeanStd, 
                        val_rms: RunningMeanStd):
        
        # normalize input if needed, we update rms only here;
        processed_obs = obs['obs']
        if self.normalize_input:
            # update rms;
            with th.no_grad():
                self.obs_rms.update(processed_obs)
            processed_obs = obs_rms.normalize(processed_obs)
        
        # [std] is a vector of length [action_dim], which is shared by all the envs;
        actions, mu, std, eps = self.actor.forward_with_dist(processed_obs, deterministic=False)
        if std.ndim == 1:
            std = std.unsqueeze(0)                      
            std = std.expand(mu.shape[0], -1).clone()      # make size of [std] same as [actions] and [mu];
        neglogp = self.neglogp(actions, mu, std, th.log(std))

        # self.target_critic.eval()
        values = self.target_critic(processed_obs)
        
        # if using normalize value, target_critic learns to give normalized state values;
        # therefore, unnormalize the resulting value;
        if self.normalize_value:
            values = val_rms.normalize(values, True)

        res_dict = {
            "obs": processed_obs,
            "actions": actions,
            "mus": mu,
            "sigmas": std,
            "neglogpacs": neglogp,
            "values": values,
            "rnn_states": None,
            'rp_eps': eps,
        }

        return res_dict

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
        Train actor based on other methods than PPO with current experience.
        '''
        
        raise NotImplementedError()

    def train_critic(self,
                        grad_obses: List[th.Tensor],
                        grad_actions: List[th.Tensor],
                        grad_values: List[th.Tensor],
                        grad_next_values: List[th.Tensor],
                        grad_rewards: List[th.Tensor],
                        grad_fdones: List[th.Tensor],
                        last_fdones: th.Tensor):
        
        with th.no_grad():
        
            # compute advantage and add it to state value to get target values;
            curr_grad_advs = self.grad_advantages(self.tau,
                                                    grad_values,
                                                    grad_next_values, 
                                                    grad_rewards,
                                                    grad_fdones,
                                                    last_fdones)
            grad_advs = curr_grad_advs

            target_values = []
            for i in range(len(curr_grad_advs)):
                target_values.append(curr_grad_advs[i] + grad_values[i])

            th_obs = th.cat(grad_obses, dim=0)
            th_target_values = th.cat(target_values, dim=0)
            
            # update value rms here once;
            if self.normalize_value:
                self.val_rms.update(th_target_values)
            
            batch_size = len(th_target_values) // self.critic_num_batch
            critic_dataset = CriticDataset(batch_size, th_obs, th_target_values)

        self.critic.train()
        critic_loss = 0
        for j in range(self.critic_iterations):
            
            total_critic_loss = 0
            batch_cnt = 0
            
            for i in range(len(critic_dataset)):
            
                batch_sample = critic_dataset[i]
                self.critic_optimizer.zero_grad()

                predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
                if self.normalize_value:
                    # predicted_values = curr_val_rms.normalize(predicted_values, True)
                    predicted_values = self.val_rms.normalize(predicted_values, True)
                
                target_values = batch_sample['target_values']
                training_critic_loss = th.mean((predicted_values - target_values) ** 2, dim=0)
                training_critic_loss.backward()
                
                # ugly fix for simulation nan problem
                for params in self.critic.parameters():
                    params.grad.nan_to_num_(0.0, 0.0, 0.0)

                if self.truncate_grads:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                self.critic_optimizer.step()

                total_critic_loss += training_critic_loss
                batch_cnt += 1
            
            # critic_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
            # if self.print_stats:
            #     print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, critic_loss), end='\r')

        return grad_advs

    def update_target_critic(self):
        with th.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1. - alpha) * param.data)

    # def get_critic_values(self, obs, use_target_critic: bool, obs_rms_train: bool):

    #     if use_target_critic:   
    #         critic = self.target_critic
    #         # critic.eval()
    #     else:
    #         critic = self.critic

    #     if self.normalize_input:

    #         if obs_rms_train:
    #             self.running_mean_std.train()
    #         else:
    #             self.running_mean_std.eval()

    #     processed_obs = self._preproc_obs(obs)
    #     values = critic(processed_obs)

    #     if self.normalize_value:
    #         values = self.value_mean_std(values, True)

    #     return values

    def grad_advantages(self, gae_tau, mb_extrinsic_values, mb_next_extrinsic_values, mb_rewards, mb_fdones, last_fdones):

        num_step = len(mb_extrinsic_values)
        mb_advs = []
        
        # GAE;
        lastgaelam = 0
        for t in reversed(range(num_step)):
            if t == num_step - 1:
                nextnonterminal = 1.0 - last_fdones
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            nextvalues = mb_next_extrinsic_values[t]

            '''
            In computing delta, we do not use [nextnonterinal] because 
            [nextvalues] should be zero if the episode was finished 
            before the maximum episode length.
            
            If the episode was finished by going over horizon, we have
            to deal with the [nextvalues] that is not zero, but 
            [nextnonterminal] is still 0.

            Therefore, we do not consider [nextnonterminal] here.
            '''
            delta = mb_rewards[t] + self.gamma * nextvalues - mb_extrinsic_values[t]
            mb_adv = lastgaelam = delta + self.gamma * gae_tau * nextnonterminal * lastgaelam
            mb_advs.append(mb_adv)

        mb_advs.reverse()
        return mb_advs

    def grad_advantages_first_terms_sum(self, grad_advs, grad_start):

        num_timestep = grad_start.shape[0]
        num_actors = grad_start.shape[1]

        adv_sum = 0

        for i in range(num_timestep):
            for j in range(num_actors):
                if grad_start[i, j]:
                    adv_sum = adv_sum + grad_advs[i][j]

        return adv_sum

    def clear_experience_buffer_grads(self):

        '''
        Clear computation graph attached to the tensors in the experience buffer.
        '''

        with th.no_grad():

            for k in self.experience_buffer.tensor_dict.keys():

                if not isinstance(self.experience_buffer.tensor_dict[k], th.Tensor):

                    continue

                self.experience_buffer.tensor_dict[k] = self.experience_buffer.tensor_dict[k].detach()

    def train_actor_ppo(self, batch_dict):

        raise NotImplementedError()

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
        unnormalized_advantages = advantages

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

        return dataset_dict

        if self.gi_algorithm == "ppo-only":

            dataset_dict['mu'] = mus
            dataset_dict['sigma'] = sigmas
            dataset_dict['logp_actions'] = neglogpacs

        elif self.gi_algorithm == "grad-ppo-shac":

            with torch.no_grad():
                n_mus, n_sigmas = self.actor.forward_dist(obses)
                if n_sigmas.ndim == 1:
                    n_sigmas = n_sigmas.unsqueeze(0)                      
                    n_sigmas = n_sigmas.expand(mus.shape[0], -1).clone()
                
                n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, torch.log(n_sigmas))

                dataset_dict['mu'] = n_mus
                dataset_dict['sigma'] = n_sigmas
                dataset_dict['logp_actions'] = n_neglogpacs

        # compute [mus] and [sigmas] again here because we could have
        # updated policy in [play_steps] using RP gradients;
        # find out if the updated policy is still close enough to the
        # original policy, because PPO assumes it;
        # if it is not close enough, we decrease [alpha];
        
        elif self.gi_algorithm == "grad-ppo-alpha":
        
            with torch.no_grad():
                n_mus, n_sigmas = self.actor.forward_dist(obses)
                if n_sigmas.ndim == 1:
                    n_sigmas = n_sigmas.unsqueeze(0)                      
                    n_sigmas = n_sigmas.expand(mus.shape[0], -1).clone()
                
                n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, torch.log(n_sigmas))
                
                # find out distance between current policy and old policy;
                
                pac_ratio = torch.exp(torch.clamp(neglogpacs - n_neglogpacs, max=16.))  # prevent [inf];
                out_of_range_pac_ratio = torch.logical_or(pac_ratio < (1. - self.e_clip), 
                                                          pac_ratio > (1. + self.e_clip))
                out_of_range_pac_ratio = torch.count_nonzero(out_of_range_pac_ratio) / actions.shape[0]
                
                self.writer.add_scalar("info_alpha/oor_pac_ratio", out_of_range_pac_ratio, self.epoch_num)
                        
                # find out if current policy is better than old policy in terms of lr gradients;
                
                est_curr_performance = torch.sum(unnormalized_advantages * pac_ratio) - torch.sum(unnormalized_advantages)
                # est_curr_performance = torch.sum(advantages * pac_ratio) - torch.sum(advantages)
                
                n_est_curr_performance = self.est_curr_performace_rms.normalize(est_curr_performance)
                self.est_curr_performace_rms.update(est_curr_performance.unsqueeze(0))
                
                self.writer.add_scalar("info_alpha/est_curr_performance", est_curr_performance, self.epoch_num)
                self.writer.add_scalar("info_alpha/est_curr_performance_n", n_est_curr_performance, self.epoch_num)
                
                # if current policy is too far from old policy or is worse than old policy,
                # decrease alpha;
                
                if out_of_range_pac_ratio > self.gi_max_dist_rp_lr or \
                    (est_curr_performance < 0 and n_est_curr_performance < -1.):
                    
                    self.next_alpha = self.gi_curr_alpha / self.gi_update_factor
                    if self.gi_dynamic_alpha_scheduler in ['dynamic0', 'dynamic2']:
                        self.next_actor_lr = self.actor_lr / self.gi_update_factor
                    self.next_alpha = np.clip(self.next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                
                dataset_dict['mu'] = n_mus
                dataset_dict['sigma'] = n_sigmas
                dataset_dict['logp_actions'] = n_neglogpacs
                    
        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            raise NotImplementedError()

    # def get_full_state_weights(self):
        
    #     state = super().get_full_state_weights()

    #     state['gi_actor'] = self.actor.state_dict()
    #     state['gi_critic'] = self.critic.state_dict()
    #     state['gi_target_critic'] = self.target_critic.state_dict()
    #     if self.normalize_input:
    #         state['gi_obs_rms'] = self.obs_rms        
    #     return state

    # def set_full_state_weights(self, weights):
        
    #     super().set_full_state_weights(weights)

    #     self.actor.load_state_dict(weights['gi_actor'])
    #     self.critic.load_state_dict(weights['gi_critic'])
    #     self.target_critic.load_state_dict(weights['gi_target_critic'])
    #     if self.normalize_input:
    #         self.obs_rms = weights['gi_obs_rms'].to(self.ppo_device)
    
    # def calc_gradients(self, input_dict):

    #     # =================================================

    #     value_preds_batch = input_dict['old_values']
    #     advantage = input_dict['advantages']
    #     actions_batch = input_dict['actions']
    #     obs_batch = input_dict['obs']

    #     # these old mu and sigma are used to compute new policy's KL div from
    #     # the old policy, which could be used to update learning rate later;
    #     # it is not directly involved in policy updates;
    #     old_mu_batch = input_dict['mu']
    #     old_sigma_batch = input_dict['sigma']
        
    #     if self.gi_algorithm == "grad-ppo-alpha":
    #         old_action_log_probs_batch_0 = input_dict['old_logp_actions']       # action log probs before alpha update;
    #         old_action_log_probs_batch_1 = input_dict['logp_actions']           # action log probs after alpha update;
    #     else:
    #         old_action_log_probs_batch = input_dict['old_logp_actions']         # original action log probs;
        
    #     lr_mul = 1.0
    #     curr_e_clip = lr_mul * self.e_clip

    #     if self.is_rnn:
    #         raise NotImplementedError()
        
    #     for param in self.actor.parameters():
    #         if torch.any(torch.isnan(param.data)) or torch.any(torch.isinf(param.data)):
    #             print("Invalid param 1")
    #             exit(-1)
            
    #     # get current policy's actions;
    #     curr_mu, curr_std = self.actor.forward_dist(obs_batch)
    #     if curr_std.ndim == 1:
    #         curr_std = curr_std.unsqueeze(0)                      
    #         curr_std = curr_std.expand(curr_mu.shape[0], -1).clone()
    #     neglogp = self.neglogp(actions_batch, curr_mu, curr_std, torch.log(curr_std))
            
    #     # min_std = float(1e-5)
    #     # tmp_curr_std = curr_std
    #     # while True:
    #     #     neglogp = self.neglogp(actions_batch, curr_mu, tmp_curr_std, torch.log(tmp_curr_std))
    #     #     if torch.any(torch.isnan(neglogp)) or torch.any(torch.isinf(neglogp)):
                
    #     #         # isnan_ind = torch.isnan(neglogp)
    #     #         # isinf_ind = torch.isinf(neglogp)
    #     #         # # print(actions_batch[isnan_ind])
    #     #         # # print(curr_mu[isnan_ind])
    #     #         # # print(tmp_curr_std[isnan_ind])

    #     #         # # print(actions_batch[isinf_ind])
    #     #         # # print(curr_mu[isinf_ind])
    #     #         # # print(tmp_curr_std[isinf_ind])
                
    #     #         print(min_std)
    #     #         tmp_curr_std = torch.clamp(curr_std, min=min_std)
    #     #         min_std *= 2.
    #     #         exit(-1)
    #     #     else:
    #     #         break

    #     if self.gi_algorithm == "grad-ppo-alpha":
    #         a_loss = _grad_common_losses.actor_loss_alpha(old_action_log_probs_batch_0, 
    #                                                                    old_action_log_probs_batch_1,
    #                                                                    neglogp, 
    #                                                                    advantage, 
    #                                                                    self.ppo, 
    #                                                                    curr_e_clip)
    #     else:
    #         a_loss = _grad_common_losses.actor_loss(old_action_log_probs_batch, 
    #                                                 neglogp, 
    #                                                 advantage, 
    #                                                 self.ppo, 
    #                                                 curr_e_clip)
            
    #     c_loss = torch.zeros((1,), device=self.ppo_device)
    #     b_loss = self.bound_loss(curr_mu)

    #     # do not have entropy coef for now;
    #     losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), b_loss.unsqueeze(1)], None)
    #     a_loss, b_loss = losses[0], losses[1]

    #     entropy = torch.zeros((1,), device=self.ppo_device)
    #     assert self.entropy_coef == 0., ""

    #     # we only use actor loss here for fair comparison;
    #     loss = a_loss
        
    #     self.ppo_optimizer.zero_grad()
    #     if self.multi_gpu:
    #         raise NotImplementedError()
    #     else:
    #         for param in self.actor.parameters():
    #             param.grad = None

    #     loss.backward()
        
    #     #TODO: Refactor this ugliest code of they year
    #     if self.truncate_grads:
    #         if self.multi_gpu:
    #             raise NotImplementedError()
    #         else:
    #             nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
    #             self.ppo_optimizer.step()
    #     else:
    #         self.ppo_optimizer.step()

    #     for param in self.actor.parameters():
    #         if torch.any(torch.isnan(param.data)) or torch.any(torch.isinf(param.data)):

    #             print("Invalid param 2")
    #             print(loss)
    #             print(a_loss)
                
    #             # print(_grad_common_losses.actor_loss_alpha(old_action_log_probs_batch_0, 
    #             #                                             old_action_log_probs_batch_1,
    #             #                                             neglogp, 
    #             #                                             advantage, 
    #             #                                             self.ppo, 
    #             #                                             curr_e_clip))
    #             exit(-1)

    #     with torch.no_grad():
    #         reduce_kl = not self.is_rnn
    #         kl_dist = torch_ext.policy_kl(curr_mu.detach(), curr_std.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
    #         if self.is_rnn:
    #             raise NotImplementedError()
                
    #     self.train_result = (a_loss, c_loss, entropy, \
    #         kl_dist, self.last_lr, lr_mul, \
    #         curr_mu.detach(), curr_std.detach(), b_loss)

    # def update_lr(self, lr):
    #     if self.multi_gpu:
    #         lr_tensor = torch.tensor([lr])
    #         self.hvd.broadcast_value(lr_tensor, 'learning_rate')
    #         lr = lr_tensor.item()

    #     for param_group in self.ppo_optimizer.param_groups:
    #         param_group['lr'] = lr
            
    # def differentiate_grad_advantages(self, 
    #                                 grad_actions: torch.Tensor, 
    #                                 grad_advs: torch.Tensor, 
    #                                 grad_start: torch.Tensor,
    #                                 debug: bool=False):

    #     '''
    #     Compute first-order gradients of [grad_advs] w.r.t. [grad_actions] using automatic differentiation.
    #     '''

    #     num_timestep = grad_start.shape[0]
    #     num_actor = grad_start.shape[1]

    #     adv_sum: torch.Tensor = self.grad_advantages_first_terms_sum(grad_advs, grad_start)

    #     # compute gradients;
        
    #     # first-order gradient;
        
    #     # adv_gradient = torch.autograd.grad(adv_sum, grad_actions, retain_graph=debug)
    #     # adv_gradient = torch.stack(adv_gradient)
        
    #     for ga in grad_actions:
    #         ga.retain_grad()
    #     adv_sum.backward(retain_graph=debug)
    #     adv_gradient = []
    #     for ga in grad_actions:
    #         adv_gradient.append(ga.grad)
    #     adv_gradient = torch.stack(adv_gradient)
        
    #     # reweight grads;

    #     with torch.no_grad():

    #         c = (1.0 / (self.gamma * self.tau))
    #         cv = torch.ones((num_actor, 1), device=adv_gradient.device)

    #         for nt in range(num_timestep):

    #             # if new episode has been started, set [cv] to 1; 
    #             for na in range(num_actor):
    #                 if grad_start[nt, na]:
    #                     cv[na, 0] = 1.0

    #             adv_gradient[nt] = adv_gradient[nt] * cv
    #             cv = cv * c
                
    #     if debug:
            
    #         # compute gradients in brute force and compare;
    #         # this is to prove correctness of efficient computation of GAE-based advantage w.r.t. actions;
        
    #         for i in range(num_timestep):
                
    #             debug_adv_sum = grad_advs[i].sum()
                
    #             debug_grad_adv_gradient = torch.autograd.grad(debug_adv_sum, grad_actions[i], retain_graph=True)[0]
    #             debug_grad_adv_gradient_norm = torch.norm(debug_grad_adv_gradient, p=2, dim=-1)
                
    #             debug_grad_error = torch.norm(debug_grad_adv_gradient - adv_gradient[i], p=2, dim=-1)
    #             debug_grad_error_ratio = debug_grad_error / debug_grad_adv_gradient_norm
                
    #             assert torch.all(debug_grad_error_ratio < 0.01), \
    #                 "Gradient of advantage possibly wrong"
                        
    #     adv_gradient = adv_gradient.detach()
        
    #     return adv_gradient
    
    # def action_eps_jacobian(self, mu, sigma, eps):
        
    #     jacobian = torch.zeros((eps.shape[0], eps.shape[1], eps.shape[1]))
        
    #     for d in range(eps.shape[1]):
            
    #         if sigma.ndim == 1:
    #             jacobian[:, d, d] = sigma[d].detach()
    #         elif sigma.ndim == 2:
    #             jacobian[:, d, d] = sigma[:, d].detach()
            
    #     return jacobian
        
    #     '''
    #     distr = GradNormal(mu, sigma)
    #     eps.requires_grad = True
    #     actions = distr.eps_to_action(eps)
        
    #     jacobian = torch.zeros((eps.shape[0], actions.shape[1], eps.shape[1]))
        
    #     for d in range(actions.shape[1]):
    #         target = torch.sum(actions[:, d])
    #         grad = torch.autograd.grad(target, eps, retain_graph=True)
    #         grad = torch.stack(grad)
    #         jacobian[:, d, :] = grad
            
    #     return jacobian
    #     '''
    
    def use_ppo(self):
        '''
        Whether or not to use PPO.
        '''
        raise NotImplementedError()

    '''
    Logging
    '''
    def write_stats(self, total_time, epoch_num, step_time, no_ppo_time, ppo_time, ppo_loss, frame, scaled_time, scaled_play_time, curr_frames):
        
        mean_ppo_loss = th.tensor(ppo_loss).mean().item()

        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/no_ppo_time', no_ppo_time, frame)
        self.writer.add_scalar('performance/ppo_time', ppo_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/ppo_loss', mean_ppo_loss, frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)

        if self.use_ppo():
            self.writer.add_scalar('info/e_clip', self.e_clip, frame)
        # self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def get_weights(self):
        state = self.get_stats_weights()
        state['actor'] = self.actor.state_dict()
        state['critic'] = self.critic.state_dict()
        return state

    def get_stats_weights(self):
        state = {}
        # if self.normalize_input:
        #     state['running_mean_std'] = self.running_mean_std.state_dict()
        # if self.normalize_value:
        #     state['reward_mean_std'] = self.value_mean_std.state_dict()
        return state

    def get_optimizers_state(self):
        state = {}
        state['critic'] = self.critic_optimizer.state_dict()
        return state

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['optimizers'] = self.get_optimizers_state()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        env_state = self.vec_env.get_env_state()
        state['env_state'] = env_state

        return state
    
    def safe_filesystem_op(self, func, *args, **kwargs):
        """
        This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
        Filesystem environment (i.e. NGC cloud or SLURM)
        """
        num_attempts = 5
        for attempt in range(num_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                print(f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
                wait_sec = 2 ** attempt
                print(f'Waiting {wait_sec} before trying again...')
                time.sleep(wait_sec)

        raise RuntimeError(f'Could not execute {func}, give up after {num_attempts} attempts...')
    
    def safe_save(self, state, filename):
        return self.safe_filesystem_op(th.save, state, filename)

    def save_checkpoint(self, filename, state):
        print("=> saving checkpoint '{}'".format(filename + '.pth'))
        self.safe_save(state, filename + '.pth')
    
    def save(self, fn):
        state = self.get_full_state_weights()
        self.save_checkpoint(fn, state)