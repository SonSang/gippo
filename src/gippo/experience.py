'''
Modified from
https://github.com/Denys88/rl_games/blob/master/rl_games/common/experience.py#L285
'''

import numpy as np
import torch as th
import gym

from gippo.utils import numpy_to_torch_dtype_dict

class ExperienceBuffer:
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']
        
        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        batch_size = self.num_actors * self.num_agents
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        if type(self.action_space) is gym.spaces.Discrete:
            raise ValueError()
        if type(self.action_space) is gym.spaces.Tuple:
            raise ValueError()
        if type(self.action_space) is gym.spaces.Box:
            self.actions_shape = (self.action_space.shape[0],) 
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        state_base_shape = self.state_base_shape

        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        
        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.uint8), obs_base_shape)
        
        assert self.is_continuous, "Only continuous action space is supported"
        self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
        self.tensor_dict['mus'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
        self.tensor_dict['sigmas'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)

        '''
        Gradient info
        '''
        # store first and second order analytical gradients of advantage w.r.t. actions;
        
        base_shape = self.obs_base_shape
        action_shape = self.actions_shape
        dtype = th.float32
        device = self.device
        
        self.tensor_dict['adv_gradient'] = th.zeros(base_shape + action_shape, dtype=dtype, device=device)
        self.tensor_dict['adv_hessian'] = th.zeros(base_shape + action_shape + action_shape, dtype=dtype, device=device)

    def _init_from_aux_dict(self, tensor_dict):
        obs_base_shape = self.obs_base_shape
        for k,v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(v), dtype=np.float32), obs_base_shape)

    def _create_tensor_from_space(self, space, base_shape):       
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return th.zeros(base_shape + space.shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Discrete:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return th.zeros(base_shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Tuple:
            '''
            assuming that tuple is only Discrete tuple
            '''
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return th.zeros(base_shape +(tuple_len,), dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Dict:
            t_dict = {}
            for k,v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_dict[name][k][index,:] = v
        else:
            self.tensor_dict[name][index,:] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        
        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        
        return res_dict