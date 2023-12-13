'''
Modified from
https://github.com/Denys88/rl_games/blob/master/rl_games/common/vecenv.py
https://github.com/NVlabs/DiffRL/blob/main/examples/train_rl.py#L52
'''
vecenv_config = {}      # vectorized environment, which usually wraps around
                        # a single environment and provides parallelized interface;
env_config = {}         # single environment config;

def register_vecenv_config(config_name, func):
    vecenv_config[config_name] = func

def register_env_config(env_name, config):
    env_config[env_name] = config

def create_vecenv(env_name, num_actors, **kwargs):
    vecenv_name = env_config[env_name]['vecenv_type']
    return vecenv_config[vecenv_name](env_name, num_actors, **kwargs)

'''
Vectorized Environment
'''

class IVecEnv:
    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        pass

    def seed(self, seed):
        pass

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass

class BaseVecEnv(IVecEnv):
    def __init__(self, env_name, num_actors, **kwargs):
        kwargs['num_envs'] = num_actors
        self.env = env_config[env_name]['env_creator'](**kwargs)

        self.full_state = {}
        self.device = kwargs['device']

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.device)
        
    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(actions.to(self.device))

        return self.full_state["obs"].to(self.device), \
                reward.to(self.device), \
                is_done.to(self.device), \
                info

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.device)

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        return info

class RLGPUEnv(IVecEnv):
    def __init__(self, env_name, num_actors, **kwargs):
        self.env = env_config[env_name]['env_creator'](**kwargs)

        self.full_state = {}
        raise NotImplementedError()
        self.rl_device = "cuda:0"

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(actions.to(self.env.device))

        return self.full_state["obs"].to(self.rl_device), reward.to(self.rl_device), is_done.to(self.rl_device), info

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.rl_device)

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        print(info['action_space'], info['observation_space'])

        return info