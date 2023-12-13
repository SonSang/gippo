import numpy as np
import torch as th
import os

import matplotlib.pyplot as plt

from envs.func_optim.base import FuncOptimEnv

class DejongEnv(FuncOptimEnv):

    def __init__(self, 
                num_envs,
                dim=1, 
                seed=0, 
                no_grad=True, 
                render=False, 
                device='cuda:0'):

        super(DejongEnv, self).__init__(
            num_envs=num_envs,
            dim=dim,
            seed=seed,
            no_grad=no_grad,
            render=render,
            device=device)
        
        self.bound = 5.12
    
    def preprocess_actions(self, actions: th.Tensor):
        actions = super().preprocess_actions(actions)
        actions = actions * self.bound
        return actions

    def render(self, mode = 'human', actions = None, p_actions = None):

        if self.visualize:

            assert self.dim == 1, ""

            min_action = -self.bound
            max_action = self.bound
            step = (max_action - min_action) / self.render_resolution

            x = th.arange(min_action, max_action, step).unsqueeze(-1)
            y = self.evaluate(x)

            x = x[:, 0].cpu().numpy()
            y = y.cpu().numpy()

            f = plt.figure()
            f.set_figwidth(6.4 * 2)
            f.set_figheight(4.8 * 2)

            plt.plot(x, y, color='blue')

            with th.no_grad():

                if actions == None:
                    x = self.actions[:, 0].cpu().numpy()
                    y = self.rew_buf.cpu().numpy()
                elif actions != None:
                    x = th.clip(actions, -1, 1) * self.bound
                    y = self.evaluate(x)

                    x = x[:, 0].cpu().numpy()
                    y = y.cpu().numpy()
                else:
                    raise ValueError()

            plt.plot(x, y, 'x', color='black', markersize=5e-0)

            with th.no_grad():

                if p_actions != None:
                    x = th.clip(p_actions, -1, 1) * self.bound
                    y = self.evaluate(x)

                    x = x[:, 0].cpu().numpy()
                    y = y.cpu().numpy()

            plt.plot(x, y, 'o', color='red', markersize=2e-0)

            plt.title("Dejong Function, Step {}".format(self.num_frames))
            plt.xlabel("x")
            plt.ylabel("y")

            dir = './outputs/dejong/'

            if not os.path.exists(dir):
                os.makedirs(dir)

            plt.savefig("./outputs/dejong/dejong_{}.png".format(self.num_frames))

    def reset(self, env_ids=None, force_reset=True):
        
        self.calculateObservations()

        return self.obs_buf

    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self):
        
        pass

    '''
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):

        self.obs_buf = th.zeros_like(self.obs_buf)

    def calculateReward(self):

        self.rew_buf = self.evaluate(self.actions)

        # reset agents
        self.reset_buf = th.where(self.progress_buf > self.episode_length - 1, th.ones_like(self.reset_buf), self.reset_buf)

    def evaluate(self, x: th.Tensor):

        y = th.sum(x * x, dim=1)

        return -y