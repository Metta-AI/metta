#!/usr/bin/env python3
"""Comparison of YAML-based configuration vs direct Python usage.

This shows the transformation from the old Hydra/YAML approach
to the new library-style approach.
"""


def yaml_based_approach():
    """The OLD way - using YAML files and Hydra."""
    print("=== OLD APPROACH: YAML + Hydra ===\n")

    print("1. Create multiple YAML files:")
    print("""
    configs/
    ├── common.yaml
    ├── train_job.yaml
    ├── agent/
    │   └── simple.yaml
    ├── trainer/
    │   └── puffer.yaml
    ├── sim/
    │   └── all.yaml
    └── wandb/
        └── metta_research.yaml
    """)

    print("2. Write complex nested configurations:")
    print("""
    # train_job.yaml
    defaults:
      - common
      - agent: simple
      - trainer: puffer
      - sim: all
      - wandb: metta_research
      - _self_

    seed: 1
    cmd: train
    """)

    print("3. Agent defined with Hydra targets:")
    print("""
    # agent/simple.yaml
    agent:
      _target_: metta.agent.metta_agent.MettaAgent
      components:
        _obs_:
          _target_: metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper
        cnn1:
          _target_: metta.agent.lib.nn_layer_library.Conv2d
          sources: [{name: obs_normalizer}]
          nn_params:
            out_channels: 64
            kernel_size: 5
    """)

    print("4. Run with Hydra CLI:")
    print("""
    python train.py --config-path configs --config-name train_job
    """)

    print("\nProblems:")
    print("- Scattered configuration across many files")
    print("- Opaque _target_ syntax")
    print("- Hard to debug and understand")
    print("- No IDE support or type checking")
    print("- Complex override syntax")


def python_based_approach():
    """The NEW way - direct Python usage."""
    print("\n\n=== NEW APPROACH: Direct Python ===\n")

    print("1. Import what you need:")
    print("""
    from metta import SimpleCNNAgent, Metta, configure
    """)

    print("2. Create objects directly:")
    print("""
    # Configure runtime (replaces common.yaml)
    runtime = configure(
        run_name="my_experiment",
        device="cuda",
        seed=1
    )

    # Create agent (replaces agent/simple.yaml)
    agent = SimpleCNNAgent(
        obs_width=11,
        obs_height=11,
        hidden_size=256,
        lstm_layers=3
    )

    # Create trainer (replaces trainer/puffer.yaml + train_job.yaml)
    metta = Metta(
        agent=agent,
        total_timesteps=10_000_000,
        batch_size=32768,
        learning_rate=3e-4
    )
    """)

    print("3. Train:")
    print("""
    metta.train()
    """)

    print("\nBenefits:")
    print("- Everything in one place")
    print("- Clear, readable Python code")
    print("- Full IDE support with autocomplete")
    print("- Easy to debug")
    print("- Type safety")


def feature_comparison():
    """Compare specific features between approaches."""
    print("\n\n=== FEATURE COMPARISON ===\n")

    # Custom agent
    print("Creating a custom agent:")
    print("\nOLD (YAML):")
    print("""
    # configs/agent/my_custom.yaml
    agent:
      _target_: metta.agent.metta_agent.MettaAgent
      components:
        encoder:
          _target_: my_module.CustomEncoder
          setup:
            - source: obs_normalizer
              target: input
        lstm:
          _target_: metta.agent.lib.lstm.LSTM
          setup:
            - source: encoder.output
              target: input
    """)

    print("\nNEW (Python):")
    print("""
    class MyCustomAgent(BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.encoder = CustomEncoder()
            self.lstm = nn.LSTM(256, 512, 2)

        def forward(self, x, state):
            x = self.encoder(x)
            x, state = self.lstm(x, state)
            return x, state
    """)

    # Hyperparameter sweeps
    print("\n\nHyperparameter sweeps:")
    print("\nOLD (Hydra multirun):")
    print("""
    python train.py -m agent.hidden_size=128,256,512 trainer.lr=1e-3,1e-4
    """)

    print("\nNEW (Python loop):")
    print("""
    for hidden_size in [128, 256, 512]:
        for lr in [1e-3, 1e-4]:
            agent = SimpleCNNAgent(hidden_size=hidden_size)
            metta = Metta(agent=agent, learning_rate=lr)
            metta.train()
    """)

    # Environment configuration
    print("\n\nEnvironment configuration:")
    print("\nOLD (YAML):")
    print("""
    # configs/env/large.yaml
    env:
      _target_: mettagrid.MettaGridEnv
      width: 21
      height: 21
      max_steps: 2048
      num_agents: 4
    """)

    print("\nNEW (Python):")
    print("""
    env = create_env(width=21, height=21, max_steps=2048, num_agents=4)
    # or
    env = create_env_from_preset("large")
    """)


def migration_example():
    """Show a complete migration example."""
    print("\n\n=== COMPLETE MIGRATION EXAMPLE ===\n")

    print("OLD train.py:")
    print("""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="configs", config_name="train_job")
    def train(cfg: DictConfig):
        # Complex setup with Hydra instantiation
        env = hydra.utils.instantiate(cfg.env)
        agent = hydra.utils.instantiate(cfg.agent)
        trainer = hydra.utils.instantiate(cfg.trainer)

        # Training loop buried in trainer
        trainer.train()

    if __name__ == "__main__":
        train()
    """)

    print("\n\nNEW train.py:")
    print("""
    from metta import SimpleCNNAgent, Metta, create_env

    def train():
        # Direct, clear setup
        env = create_env(width=15, height=15)
        agent = SimpleCNNAgent(hidden_size=256)

        # Transparent training
        metta = Metta(agent=agent, env=env)

        # Custom training loop if needed
        while metta.training():
            metta.train(timesteps=100_000)
            print(f"Step {metta.agent_step}: {metta.eval()}")

    if __name__ == "__main__":
        train()
    """)


def advanced_patterns():
    """Show advanced usage patterns."""
    print("\n\n=== ADVANCED PATTERNS ===\n")

    print("1. Composition over configuration:")
    print("""
    # Compose functionality as needed
    agent = AttentionAgent(num_heads=8)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    metta = Metta(
        agent=agent,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        on_checkpoint=lambda m: save_model(m.agent)
    )
    """)

    print("\n2. Dynamic experimentation:")
    print("""
    # Easy to experiment programmatically
    results = {}

    for agent_type in ["simple_cnn", "attention", "large_cnn"]:
        for hidden_size in [128, 256, 512]:
            agent = create_agent(agent_type, hidden_size=hidden_size)
            trained = train(agent, timesteps=1_000_000)
            results[f"{agent_type}_{hidden_size}"] = evaluate(trained)
    """)

    print("\n3. Integration with existing code:")
    print("""
    # Use your existing PyTorch models
    class YourExistingModel(nn.Module):
        ...

    # Wrap as Metta agent
    class WrappedAgent(BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = YourExistingModel()

    # Train normally
    agent = WrappedAgent()
    metta = Metta(agent=agent)
    metta.train()
    """)


def summary():
    """Summarize the benefits."""
    print("\n\n=== SUMMARY ===\n")

    print("The refactoring transforms Metta from a configuration-heavy")
    print("framework to a library that follows Python best practices:")
    print()
    print("✓ Direct object creation instead of YAML")
    print("✓ Clear, debuggable Python code")
    print("✓ Full IDE support with type hints")
    print("✓ Composable, modular components")
    print("✓ Easy integration with existing code")
    print("✓ Pythonic API similar to PyTorch/scikit-learn")
    print()
    print("Old: Configure everything in YAML, hope it works")
    print("New: Write Python, see exactly what happens")


if __name__ == "__main__":
    yaml_based_approach()
    python_based_approach()
    feature_comparison()
    migration_example()
    advanced_patterns()
    summary()
