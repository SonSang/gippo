'''
Modified from
https://github.com/NVlabs/DiffRL/blob/main/envs/dflex_env.py
'''
import numpy as np
import torch as th

from gym import spaces

class BaseEnv:
    
    def __init__(self, 
                num_envs, 
                num_obs, 
                num_act, 
                episode_length, 
                seed=0, 
                no_grad=True, 
                render=False, 
                device='cuda:0'):
        
        self.seed = seed

        self.no_grad = no_grad
        
        self.episode_length = episode_length

        self.device = device

        self.render = render

        self.sim_time = 0.0

        self.num_frames = 0 # record the number of frames for rendering

        self.num_environments = num_envs
        self.num_agents = 1

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act

        self.obs_space = spaces.Box(np.ones(self.num_observations, dtype=np.float32) * -np.Inf, 
                                    np.ones(self.num_observations, dtype=np.float32) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions, dtype=np.float32) * np.float32(-1.), 
                                    np.ones(self.num_actions, dtype=np.float32) * np.float32(1.))

        # allocate buffers
        self.obs_buf = th.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=th.float32, requires_grad=False)
        self.rew_buf = th.zeros(
            self.num_envs, device=self.device, dtype=th.float32, requires_grad=False)
        self.reset_buf = th.ones(
            self.num_envs, device=self.device, dtype=th.int64, requires_grad=False)
        
        # end of the episode
        self.termination_buf = th.zeros(
            self.num_envs, device=self.device, dtype=th.int64, requires_grad=False)
        self.progress_buf = th.zeros(
            self.num_envs, device=self.device, dtype=th.int64, requires_grad=False)
        self.actions = th.zeros(
            (self.num_envs, self.num_actions), device = self.device, dtype = th.float32, requires_grad = False)

        self.extras = {}

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def get_state(self):
        raise NotImplementedError()
    
    def reset_with_state(self, env_ids=None, force_reset=True):
        raise NotImplementedError()