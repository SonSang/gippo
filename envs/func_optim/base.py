import torch as th

from envs.base import BaseEnv

class FuncOptimEnv(BaseEnv):

    def __init__(self,
                num_envs,
                dim=1, 
                seed=0, 
                no_grad=True, 
                render=False, 
                device='cuda:0'):
        
        super(FuncOptimEnv, self).__init__(
            num_envs=num_envs,
            num_obs=1, 
            num_act=dim,
            episode_length=1,
            seed=seed,
            no_grad=no_grad,
            render=render,
            device=device
        )

        self.dim = dim
        self.render_resolution = 1e3
    
    def preprocess_actions(self, actions: th.Tensor):
        actions = actions.view((self.num_envs, self.num_actions))         
        actions = th.clip(actions, -1., 1.)
        return actions

    def step(self, actions: th.Tensor):
        actions = self.preprocess_actions(actions)
        self.actions = actions
            
        self.reset_buf = th.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        self.reset()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):

        self.calculateObservations()
        self.progress_buf[:] = 0
        
        return self.obs_buf

    def calculateObservations(self):

        self.obs_buf = th.zeros_like(self.obs_buf)

    def calculateReward(self):

        self.rew_buf = self.evaluate(self.actions)

        # reset agents
        self.reset_buf = th.where(self.progress_buf > self.episode_length - 1, th.ones_like(self.reset_buf), self.reset_buf)

    def evaluate(self, x: th.Tensor):

        raise NotImplementedError()