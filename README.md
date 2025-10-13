# VecEnv-SAC

A PyTorch implementation of Soft Actor-Critic (SAC) algorithm using vectorized environments for efficient training on MuJoCo continuous control tasks. This implementation focuses on scalability and performance through parallel environment simulation.

## Features

- **Soft Actor-Critic (SAC):** 
  - Maximum entropy reinforcement learning algorithm
  - Automatic temperature tuning
  - Off-policy training for sample efficiency
  - Continuous action spaces support

- **Parallel Processing:**
  - Utilizes `AsyncVectorEnv` for parallel environment simulation
  - Automatically scales to available CPU cores
  - Significantly faster training compared to sequential implementation

- **Supported Environments:**
  - [Ant-v5](SAC/Agent.py)
  - [HalfCheetah-v5](SAC/Agent.py)
  - [Hopper-v5](SAC/Agent.py)
  - [Humanoid-v5](SAC/Agent.py)
  - [HumanoidStandup-v5](SAC/Agent.py)
  - [InvertedPendulum-v5](SAC/Agent.py)
  - [InvertedDoublePendulum-v5](SAC/Agent.py)
  - [Pendulum-v1](SAC/Agent.py)
  - [Pusher-v5](SAC/Agent.py)
  - [Reacher-v5](SAC/Agent.py)
  - [Swimmer-v5](SAC/Agent.py)
  - [Walker2d-v5](SAC/Agent.py)

- **Monitoring & Visualization:**
  - Automatic video recording of evaluation episodes
  - Training curves for rewards
  - Alpha (temperature) adaptation curves
  - Policy entropy tracking

## Installation

### Environment
- Python 3.12.9
- CUDA 12.8
- PyTorch 2.7.0+cu128
- NumPy 2.2.6
- Gymnasium 1.1.1
- Matplotlib 3.10.3

```bash
# Create a new conda environment
conda create -n sac python=3.12.9
conda activate sac

# Install dependencies
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128
pip install numpy==2.2.6 matplotlib==3.10.3
pip install gymnasium[mujoco]==1.1.1
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VecEnv-SAC.git
cd VecEnv-SAC
```

2. Train an agent:
```bash
# For example, to train on Humanoid:
python Humanoid_SAC.py

# With custom hyperparameters:
python Humanoid_SAC.py --lr 0.0001 --batch_size 1024 --max_train_steps 2000000
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Learning rate |
| `batch_size` | 512 | Batch size for training |
| `buffer_size` | 1e6 | Replay buffer size |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coefficient |
| `init_alpha` | 0.2 | Initial temperature |

## Project Structure

```
.
├── SAC/
│   ├── ActorCritic.py    # Neural network architectures
│   ├── Agent.py          # SAC algorithm implementation
│   └── ReplayBuffer.py   # Experience replay buffer
├── *_SAC.py             # Environment-specific training scripts
├── Plot/                # Training visualization outputs
└── Video/              # Evaluation episode recordings
```

## Results and Visualization

- Training curves are saved in `Plot` directory:
  - Reward curves
  - Alpha adaptation
  - Policy entropy

- Evaluation videos are saved in `Video/<env_name>/`

## Contributing

Feel free to open issues or submit pull requests. Areas for improvement:

- Additional environment support
- Hyperparameter tuning
- Documentation improvements
- Performance optimizations

## References

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Documentation](https://mujoco.org/)