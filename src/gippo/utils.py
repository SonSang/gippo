import numpy as np
import torch as th
import random
import os

'''
From 
https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/utils/common.py#L72
'''
def seeding(seed=0, torch_deterministic=False):
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
        th.use_deterministic_algorithms(True)
    else:
        th.backends.cudnn.benchmark = True
        th.backends.cudnn.deterministic = False

    return seed

from torch.distributions.utils import _standard_normal
class Normal(th.distributions.Normal):
    
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args)
    
    def sample_eps(self, sample_shape=th.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return eps
    
    def eps_to_action(self, eps):
        return self.loc + eps * self.scale
    
'''
From 
https://github.com/NVlabs/DiffRL/blob/main/utils/running_mean_std.py
'''
from typing import Tuple
class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device = 'cuda:0'):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = th.zeros(shape, dtype = th.float32, device = device)
        self.var = th.ones(shape, dtype = th.float32, device = device)
        self.count = epsilon

    def to(self, device):
        rms = RunningMeanStd(device = device)
        rms.mean = self.mean.to(device).clone()
        rms.var = self.var.to(device).clone()
        rms.count = self.count
        return rms
    
    @th.no_grad()
    def update(self, arr: th.tensor) -> None:
        batch_mean = th.mean(arr, dim = 0)
        batch_var = th.var(arr, dim = 0, unbiased = False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: th.tensor, batch_var: th.tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + th.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr:th.tensor, un_norm = False) -> th.tensor:
        if not un_norm:
            result = (arr - self.mean) / th.sqrt(self.var + 1e-5)
        else:
            result = arr * th.sqrt(self.var + 1e-5) + self.mean
        return result

'''
From
https://github.com/SonSang/DiffRL/blob/stable/externals/rl_games/rl_games/algos_torch/torch_ext.py#L275
'''
class AverageMeter(th.nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", th.zeros(in_shape, dtype = th.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = th.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()

'''
From
https://github.com/Denys88/rl_games/blob/master/rl_games/common/a2c_common.py#L30
'''
def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

'''
From 
https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/torch_ext.py#L10
'''
numpy_to_torch_dtype_dict = {
    np.dtype('bool')       : th.bool,
    np.dtype('uint8')      : th.uint8,
    np.dtype('int8')       : th.int8,
    np.dtype('int16')      : th.int16,
    np.dtype('int32')      : th.int32,
    np.dtype('int64')      : th.int64,
    np.dtype('float16')    : th.float16,
    np.dtype('float32')    : th.float32,
    np.dtype('float64')    : th.float64,
    np.dtype('complex64')  : th.complex64,
    np.dtype('complex128') : th.complex128,
}