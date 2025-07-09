# Metta AI Quick Guide

A concise guide to get you started with Metta AI quickly.

## 🚀 Installation (5 minutes)

```bash
# Clone and install
git clone https://github.com/Metta-AI/metta.git
cd metta
./install.sh  # Interactive setup
```

## 🎮 First Training Run (10 minutes)

```bash
# Train a simple agent
./tools/train.py run=my_first_run +hardware=macbook wandb=off

# With GPU (if available)
./tools/train.py run=my_first_run device=cuda wandb=off
```

## 👀 Visualize Your Agent

```bash
# Interactive browser viewer
./tools/play.py run=my_first_run

# Terminal viewer
./tools/renderer.py run=my_first_run
```

## 🔧 Key Concepts

### Environment
- **MettagGrid**: Grid-based multi-agent environment
- **Actions**: Move, attack, shield, share energy, place markers
- **Resources**: Diamonds → Energy → Rewards

### Training
- **PPO**: Default reinforcement learning algorithm
- **Checkpoints**: Saved in `train_dir/<run_name>/`
- **Configs**: Compose settings with `+config_group=option`

### Configuration
```bash
# Common patterns
./tools/train.py \
    run=experiment_name \         # Name your experiment
    +hardware=macbook \          # Hardware profile
    +user=yourname \            # Personal settings
    wandb=off \                 # Disable logging
    trainer.total_timesteps=1e6 # Override parameters
```

## 📊 Using WandB

1. Set up credentials:
   ```bash
   # Add to ~/.netrc
   machine api.wandb.ai
     login user
     password YOUR_API_KEY
   ```

2. Train with logging:
   ```bash
   ./tools/train.py run=experiment wandb=external_user
   ```

## 🧪 Evaluation

```bash
# Run standard evaluations
./tools/sim.py sim=navigation run=eval_test policy_uri=./train_dir/my_first_run/checkpoints/latest.pt
```

## 📁 Project Structure

```
metta/
├── configs/        # Configuration files
├── tools/          # CLI tools (train, play, sim, etc.)
├── metta/          # Core Python package
├── mettagrid/      # C++ environment
├── docs/           # Documentation
└── train_dir/      # Training outputs
```

## 🔍 Next Steps

1. **Explore configs**: Check `configs/` for different environments and agents
2. **Read the API docs**: [docs/api.md](./api.md) for programmatic usage
3. **Join Discord**: <https://discord.gg/mQzrgwqmwy>
4. **Try different environments**:
   ```bash
   # Navigation challenges
   ./tools/train.py run=nav_test +sim=navigation

   # Multi-agent scenarios
   ./tools/train.py run=multi_test env.curriculum.num_agents=8
   ```

## 💡 Tips

- Start with short runs (`trainer.total_timesteps=1e6`)
- Use `wandb=off` for local experiments
- Add `+hardware=github` for minimal resource usage
- Check `./train_dir/<run>/` for outputs and logs

## 🆘 Common Issues

**Import errors**: Run `metta status` to check installation

**CUDA errors**: Use `device=cpu` or check GPU setup

**Out of memory**: Reduce `trainer.batch_size` or `env.num_envs`

**Slow training**: Enable `trainer.use_kickstarter=true`

---

Ready to dive deeper? Check the [full documentation](./README.md) or [API reference](./api.md).
