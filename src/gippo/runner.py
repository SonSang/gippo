import os
import time
import random
from copy import deepcopy

import numpy as np
import torch as th

from gippo.rl_algorithm.lr import LR
from gippo.rl_algorithm.rp import RP
from gippo.rl_algorithm.lrp import LRP
from gippo.rl_algorithm.ppo import PPO
from gippo.rl_algorithm.gippo import GIPPO
from gippo.vecenv import create_vecenv

class Runner:

    def __init__(self):
        th.backends.cudnn.benchmark = True
        
    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            th.manual_seed(self.seed)
            th.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'config' in params['env']:
                params['env']['config']['seed'] = self.seed
                
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')

        algo_config = self.params['algo']
        env_config = self.params['env']
        device = self.params['device']
        log_path = self.params['log_path']
        
        if self.algo_name == 'lr':
            agent = LR(algo_config, env_config, device, log_path)
        elif self.algo_name == 'rp':
            agent = RP(algo_config, env_config, device, log_path)
        elif self.algo_name == 'lrp':
            agent = LRP(algo_config, env_config, device, log_path)
        elif self.algo_name == 'ppo':
            agent = PPO(algo_config, env_config, device, log_path)
        elif self.algo_name == 'gippo':
            agent = GIPPO(algo_config, env_config, device, log_path)
        else:
            raise NotImplementedError()
        # _restore(agent, args)
        # _override_sigma(agent, args)
        agent.train()