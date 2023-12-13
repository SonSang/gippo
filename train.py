import os
import argparse

import os
import yaml
import time

from gippo import vecenv
from gippo.runner import Runner

from envs.func_optim.dejong import DejongEnv
from envs.func_optim.ackley import AckleyEnv

vecenv.register_vecenv_config(
                'BASE',
                lambda env_name,
                num_actors,
                **kwargs: vecenv.BaseVecEnv(env_name, num_actors, **kwargs))

vecenv.register_env_config(
    'DejongEnv',
    {
        'vecenv_type': 'BASE',
        'env_creator': lambda **kwargs: DejongEnv(**kwargs),
    }
)
vecenv.register_env_config(
    'AckleyEnv',
    {
        'vecenv_type': 'BASE',
        'env_creator': lambda **kwargs: AckleyEnv(**kwargs)
    }
)

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    return args

def get_args():
    custom_parameters = [
        {"name": "--cfg", "type": str, "default": "./config/func_optim/dejong/lr.yaml",
            "help": "Configuration file for training"},
        {"name": "--device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."},
        {"name": "--logdir", "type": str, "default": "logdir/"},
        {"name": "--seed", "type": int, "default": 1},]

    # parse arguments
    args = parse_arguments(
        description="Training args",
        custom_parameters=custom_parameters)
    
    return args

if __name__ == '__main__':

    args = get_args()
    vargs = vars(args)
    
    with open(args.cfg, 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)
    
    # save command line args to config;
    cfg_train["params"]["command_line_args"] = {}
    for key in vargs.keys():
        cfg_train["params"]["command_line_args"][key] = vargs[key]

    # save config;
    log_dir = cfg_train["params"]["command_line_args"]["logdir"]
    log_dir = log_dir + time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(log_dir, exist_ok = True)
    yaml.dump(cfg_train, open(os.path.join(log_dir, 'cfg.yaml'), 'w'))
    cfg_train["params"]["log_path"] = log_dir
    cfg_train["params"]["device"] = vargs["device"]
    cfg_train["params"]["seed"] = vargs["seed"]

    runner = Runner()
    runner.load(cfg_train)
    runner.run_train(vargs)