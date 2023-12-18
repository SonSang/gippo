'''
Modified from 
https://github.com/NVlabs/DiffRL/blob/main/models/actor.py
https://github.com/NVlabs/DiffRL/blob/main/models/critic.py
https://github.com/NVlabs/DiffRL/blob/main/models/model_utils.py
'''
import numpy as np
import torch as th
from torch import nn

from gippo.utils import Normal

'''
Initialize the parameters of module using the given weight and bias initialization functions.
'''
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data) #, gain=gain)
    bias_init(module.bias.data)
    return module

def get_activation_func(activation_name):
    if activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError('Activation func {} not defined'.format(activation_name))

'''
Actor
'''
class ActorStochasticMLP(nn.Module):
    def __init__(self, 
                obs_dim, 
                action_dim, 
                cfg_network, 
                device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device
        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units']

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            modules.append(get_activation_func(cfg_network['actor_mlp']['activation']))
            modules.append(th.nn.LayerNorm(self.layer_dims[i + 1]))
        self.actor_mlp = nn.Sequential(*modules).to(device)
        
        # mu;
        out_size = self.layer_dims[-1]
        self.mu = [nn.Linear(out_size, action_dim), get_activation_func('identity')]
        self.mu = nn.Sequential(*self.mu).to(device)
        
        # logstd;
        self.fixed_sigma = cfg_network['fixed_sigma']
        if cfg_network['fixed_sigma']:
            logstd = cfg_network.get('actor_logstd_init', -1.0)
            self.logstd = nn.Parameter(th.ones(action_dim, dtype=th.float32, device=device) * logstd)
        else:
            self.logstd = nn.Linear(out_size, action_dim).to(device)
            
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # print(self.actor_mlp)
        # print(self.mu)
        # print(self.logstd)

    def forward(self, obs, deterministic = False):
        out = self.actor_mlp(obs)
        mu = self.mu(out)

        if deterministic:
            return mu
        else:
            if self.fixed_sigma:
                std = self.logstd.exp() # (num_actions)
            else:
                std = th.exp(self.logstd(out))
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):
        mu, std = self.forward_dist(obs)
            
        dist = Normal(mu, std)
        eps = dist.sample_eps()
        
        if deterministic:
            eps = eps.zero_()
        sample = dist.eps_to_action(eps)

        return sample, mu, std, eps
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu, std = self.forward_dist(obs)    
        dist = Normal(mu, std)
        return dist.log_prob(actions)

    def forward_dist(self, obs):
        out = self.actor_mlp(obs)
        mu = self.mu(out)
        if self.fixed_sigma:
            std = self.logstd.exp() # (num_actions)
        else:
            std = th.exp(self.logstd(out))
            
        return mu, std
    
'''
Critic
'''
class CriticMLP(nn.Module):
    def __init__(self, obs_dim, cfg_network, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['critic_mlp']['units'] + [1]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(get_activation_func(cfg_network['critic_mlp']['activation']))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)
    
        self.obs_dim = obs_dim

        # print(self.critic)

    def forward(self, observations):
        return self.critic(observations)
