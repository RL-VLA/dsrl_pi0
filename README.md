<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation

> **Note on JAX / CUDA:** the original repo pins `jax==0.5.0` with the CUDA 12
> plugin. That stack does **not** support Blackwell GPUs (RTX 5090, sm_120).
> This fork is pinned to `jax==0.10.0` + `jax-cuda13-plugin==0.10.0` and pulls in
> newer `flax`, `orbax-checkpoint`, `wandb`, `jaxtyping`, `optax`, and `distrax`
> to keep the API surface consistent. It is backward-compatible on Hopper /
> Ada GPUs as well — the CUDA 13 plugin is runtime-compatible with any recent
> NVIDIA driver.

1. Create a Python 3.11 virtual environment. We use [uv](https://docs.astral.sh/uv/):
```
uv venv --python 3.11
source .venv/bin/activate
```
(A `conda create -n dsrl_pi0 python=3.11.11` env works too — just replace
every `uv pip install` below with `pip install`.)

2. Clone this repo with all submodules:
```
git clone git@github.com:RL-VLA/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies:
```
uv pip install -e .
uv pip install -r requirements.txt

# JAX 0.10 with the CUDA 13 plugin (Blackwell-capable)
uv pip install --upgrade "jax==0.10.0" "jaxlib==0.10.0" "jax-cuda13-plugin==0.10.0"

# Pulled in by the jax bump — versions verified to work together
uv pip install --upgrade "flax==0.12.6" "orbax-checkpoint==0.11.36" \
  "wandb==0.26.0" "jaxtyping==0.3.9" "optax==0.2.8" "distrax==0.1.8"

# install openpi
uv pip install -e openpi
uv pip install -e openpi/packages/openpi-client

# install Libero
uv pip install -e LIBERO
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero

# First-time libero setup: regenerate ~/.libero/config.yaml with local paths
uv run python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

4. **If you have multiple CUDA plugins installed** (e.g. you previously ran the
   original `jax[cuda12]==0.5.0` install), remove the cu12 artifacts to avoid
   `PJRT_Api already exists for device type cuda`:
```
uv pip uninstall -y jax-cuda12-pjrt jax-cuda12-plugin
```

5. Sanity check:
```
uv run python -c "import jax; print(jax.devices())"
# → [CudaDevice(id=0), ...]
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```

Before running, edit the `WANDB_ENTITY` line in the script (top of
`examples/scripts/run_*.sh`) to your own entity.

See [`docs/training_loop.md`](docs/training_loop.md) for a walkthrough of the
training loop (pi0 × SAC interaction, replay buffer, reward relabeling) and
[`docs/config_params.md`](docs/config_params.md) for the full list of CLI
flags and their Libero defaults.

### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
