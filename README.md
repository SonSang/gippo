# Introduction

This is the code repository for the paper ["Gradient Informed Proximal Policy Optimization"](https://arxiv.org/abs/2312.08710), which was presented in the Neurips 2023 conference. This code was implemented on the basis of [rl_games](https://github.com/Denys88/rl_games) and [SHAC](https://github.com/NVlabs/DiffRL).

# Installation

We need following packages.

* pytorch 1.13.1 (https://pytorch.org/get-started/previous-versions/)
* pyyaml 6.0.1 (pip install pyyaml)
* tensorboard (pip install tensorboard)
* tensorboardx 2.6.2 (pip install tensorboardx)
* urdfpy (pip install urdfpy)
* usd-core 23.8 (pip install usd-core)
* ray 2.6.2 (pip install ray)
* ninja 1.10.2 (conda install -c conda-forge ninja)
* cudatoolkit (conda install -c anaconda cudatoolkit)
* cudatoolkit-dev (conda install -c conda-forge cudatoolkit-dev)
* optuna 3.2.0 (pip install optuna)
* optuna-dashboard 0.11.0 (pip install optuna-dashboard)
* matplotlib (pip install matplotlib)
* highway-env 1.8.2 (pip install highway-env)
* seaborn (pip install seaborn)
* gym (pip install gym)

Then, run following command.

```bash
pip install -e .
```

# Usage

Run following command for function optimization problems.

```bash
bash ./run_func_optim.sh
```
