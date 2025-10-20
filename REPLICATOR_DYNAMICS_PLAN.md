# Replicator Dynamics Training Implementation Plan

## Overview
Replace gradient-based PPO with evolutionary replicator dynamics where:
- **Genome**: 16D latent vector representing an agent
- **Development**: Diffusion model generates policy weights + LSTM states from genome
- **Evolution**: Population of ~100 genomes evolves via replicator equation in latent space
- **Selection**: Fitness = episode reward drives distribution updates

## Architecture Components

### 1. Genome-Conditioned Diffusion Model (`metta/rl/replicator/diffusion_hypernetwork.py`)

**Purpose**: Generate policy weights and LSTM initial states from genome

```python
class DiffusionHypernetwork:
    genome_dim: 16
    T_steps: 10  # Diffusion refinement steps

    def generate_weights(genome: Tensor[16]) -> PolicyWeights:
        """
        Input: genome [16]
        Output: All ViT-LSTM weights [~millions of params]

        Process:
        1. Start with noise ~ N(0, I) at t=T
        2. Condition on genome at each step
        3. Iteratively denoise for t=T...1
        4. Output: structured dict matching ViT-LSTM architecture
        """

    def generate_lstm_states(genome: Tensor[16], batch_size: int) -> Tuple[h, c]:
        """
        Generate initial LSTM hidden states
        Could be deterministic function of genome or also diffusion-based
        """
```

**Architecture Decisions**:
- **Backbone**: U-Net style transformer that takes [noise_weights, genome_embedding, timestep_t]
- **Conditioning**: Cross-attention on genome embedding at each denoising step
- **Output structure**: Generate weights layer-by-layer or all-at-once
  - **Recommended**: Generate per-layer to reduce memory, sequential generation
- **LSTM states**: Separate small network or final diffusion output

**Training**: Parameters of this diffusion model are what we optimize via co-evolution

---

### 2. Latent Distribution (`metta/rl/replicator/latent_distribution.py`)

**Purpose**: Represent p(x) over 16D genome space

```python
class LatentDistribution:
    """
    Represents probability distribution p(x) over genome space
    Updated via replicator equation: dp/dt = (F(x) - F_bar) * p(x) + diffusion_term
    """

    def __init__(self, genome_dim: 16):
        # DECISION NEEDED: Representation choice
        pass

    def sample(self, n: int) -> Tensor[n, 16]:
        """Sample n genomes from current distribution"""

    def update(self, genomes: Tensor[n, 16], fitness: Tensor[n], dt: float):
        """Update distribution via Euler step of replicator dynamics"""

    def entropy(self) -> float:
        """Monitor diversity of distribution"""
```

**Representation Options** (DECISION NEEDED):

**Option A: Gaussian Mixture Model** (Recommended)
- Pro: Continuous, differentiable, easy to sample
- Pro: Natural diffusion term (covariance)
- Con: May struggle with multimodal fitness landscapes
- Implementation: K mixtures (K~5-10), update mixture weights and means/covs

**Option B: Normalizing Flow**
- Pro: Very flexible distribution family
- Pro: Exact likelihood computation
- Con: More complex, harder to ensure stability
- Implementation: Use neural spline flows

**Option C: Particle-Based (Empirical)**
- Pro: Simplest, no parametric assumptions
- Pro: Naturally handles multimodality
- Con: Requires kernel density estimation for p(x)
- Con: Need resampling strategy
- Implementation: Maintain weighted particle set, KDE for density

**Recommendation**: Start with **Gaussian Mixture (Option A)** for stability and ease of monitoring.

---

### 3. Fitness Estimator (`metta/rl/replicator/fitness_estimator.py`)

**Purpose**: Maintain Bayesian estimate of F(x) over genome space

```python
class FitnessEstimator:
    """
    Bayesian model: F(x) ~ GP(mean, kernel) or simpler parametric model
    """

    def __init__(self, genome_dim: 16):
        # DECISION NEEDED: GP vs neural network surrogate
        pass

    def update(self, genome: Tensor[16], fitness: float):
        """Update fitness model with new observation"""

    def predict(self, genome: Tensor[16]) -> Tuple[mean, std]:
        """Predict fitness for genome (with uncertainty)"""

    def expected_fitness(self, distribution: LatentDistribution) -> float:
        """Compute F_bar = E_p[F(x)]"""
```

**Options** (DECISION NEEDED):

**Option A: Gaussian Process** (Traditional for Bayesian optimization)
- Pro: Principled uncertainty quantification
- Pro: Works with uniform prior naturally
- Con: Scales poorly (O(n³) for n samples)
- Con: Need to choose kernel (RBF, Matern)
- Implementation: Use GPyTorch with inducing points for scalability

**Option B: Neural Surrogate Model**
- Pro: Scales to many observations
- Pro: Can learn complex fitness landscapes
- Con: Uncertainty estimates less principled
- Implementation: Ensemble of MLPs with dropout

**Option C: Local Linear Model**
- Pro: Very simple and stable
- Pro: Natural for local exploitation
- Con: Poor for global optimization
- Implementation: Fit linear model on recent K observations

**Recommendation**:
- Start with **Option B (Neural Surrogate)** - simple ensemble of 3-5 small MLPs
- Input: genome [16], Output: fitness scalar
- Train with recent history buffer (last 1000 evaluations)
- Use ensemble disagreement as uncertainty proxy

---

### 4. Population Manager (`metta/rl/replicator/population_manager.py`)

**Purpose**: Manage population sampling and fitness collection

```python
class PopulationManager:
    """
    Manages population of genomes and their fitness evaluations
    """

    def __init__(self,
                 population_size: int = 100,
                 diffusion_hypernetwork: DiffusionHypernetwork,
                 latent_distribution: LatentDistribution):
        self.population_size = population_size
        self.hypernetwork = diffusion_hypernetwork
        self.distribution = latent_distribution

    def sample_population(self) -> List[Tuple[genome, policy_weights, lstm_states]]:
        """
        Sample population_size genomes from p(x)
        Generate policy weights for each via diffusion model
        """
        genomes = self.distribution.sample(self.population_size)

        population = []
        for genome in genomes:
            weights = self.hypernetwork.generate_weights(genome)
            lstm_states = self.hypernetwork.generate_lstm_states(genome, batch_size)
            population.append((genome, weights, lstm_states))

        return population

    def collect_fitness(self,
                        population: List[genome],
                        vec_env: VectorizedEnvironment) -> Tensor[population_size]:
        """
        Evaluate population across vectorized environments (Option A approach)

        Strategy:
        1. Split vec_env into chunks of ~10 envs per genome
        2. Evaluate each genome across its chunk
        3. Average fitness across chunk
        4. Repeat for all genomes (may take multiple batches)
        """
```

**Option A Implementation Details**:
- If we have 1000 vec_envs and 100 genomes → 10 envs per genome
- Run each genome for one episode across its env subset
- Average rewards to get fitness
- This provides some noise reduction in fitness estimates

---

### 5. Replicator Update Rule (`metta/rl/replicator/replicator_equation.py`)

**Purpose**: Update p(x) via replicator dynamics

```python
def replicator_update(
    distribution: LatentDistribution,
    genomes: Tensor[N, 16],
    fitness: Tensor[N],
    dt: float = 0.1,
    diffusion_coeff: float = 0.01
) -> LatentDistribution:
    """
    Continuous-time replicator equation with diffusion:

    dp/dt = (F(x) - F_bar) * p(x) + D * ∇²p(x)

    Where:
    - F(x) = fitness at genome x
    - F_bar = E_p[F(x)] = average fitness
    - D = diffusion coefficient

    Euler step:
    p(x, t+dt) = p(x, t) + dt * [(F(x) - F_bar) * p(x, t) + D * ∇²p(x, t)]
    """

    # Implementation depends on distribution representation

    if isinstance(distribution, GaussianMixture):
        # Update mixture component weights based on fitness
        # Add diffusion by increasing covariance slightly
        ...

    elif isinstance(distribution, ParticleDistribution):
        # Reweight particles
        # Add noise for diffusion
        # Resample if needed
        ...
```

**Key Parameters** (NEED TUNING):
- `dt`: Euler step size (default: 0.1)
- `diffusion_coeff`: Maintains diversity, prevents collapse to single mode
- Update frequency: Every K episodes or every N fitness evaluations?

---

### 6. Diffusion Model Training (`metta/rl/replicator/hypernetwork_trainer.py`)

**Purpose**: Train the genome-conditioned diffusion model parameters

**Critical Challenge**: How to get gradients for the diffusion model?

The fitness landscape is:
```
genome -> [diffusion model] -> policy weights -> [vec_env rollouts] -> fitness
```

**Training Strategies** (DECISION NEEDED):

**Strategy A: Evolution Strategies Gradient Estimation**
- Perturb diffusion model parameters θ
- Evaluate fitness change
- Estimate gradient: ∇_θ E[fitness] ≈ (1/σ²) E[(fitness - baseline) * ε] where ε ~ N(0, σ²I)
- Update: θ ← θ + α * gradient_estimate
- Pro: Works without backprop through environment
- Con: High variance, needs many samples

**Strategy B: REINFORCE-style Policy Gradient**
- Treat fitness as reward signal
- Use log-likelihood trick: ∇_θ E[fitness] = E[fitness * ∇_θ log p_θ(weights|genome)]
- The diffusion model defines p_θ(weights|genome)
- Pro: Established method, less variance than ES
- Con: Still need baseline and many samples

**Strategy C: Pathwise Gradients (if possible)**
- If we can backprop through environment (or use value function surrogate)
- Directly backprop fitness → policy outputs → diffusion model parameters
- Pro: Low variance, fast learning
- Con: Requires differentiable environment or learned value model

**Strategy D: Quality-Diversity + Meta-Learning**
- Don't update diffusion model based on fitness directly
- Instead: Maximize diversity of genomes → phenotypes mapping
- Use meta-objective: "does the diffusion model produce diverse, functional policies?"
- Train with reconstruction loss + behavior diversity
- Pro: More stable, avoids local optima
- Con: Less directly optimized for fitness

**Recommendation**:
Start with **Strategy B (REINFORCE)** because:
1. Diffusion model naturally has log-likelihood
2. We can compute ∇_θ log p_θ(weights|genome) via denoising objective
3. Baseline can be running average fitness
4. Can batch updates across population

**Implementation**:
```python
class HypernetworkTrainer:
    def __init__(self, hypernetwork: DiffusionHypernetwork):
        self.hypernetwork = hypernetwork
        self.optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=1e-4)
        self.baseline_fitness = 0.0

    def update(self,
               genomes: Tensor[N, 16],
               fitness: Tensor[N]):
        """
        REINFORCE update for diffusion model parameters
        """
        # Compute log probability of generated weights under current model
        log_probs = []
        for genome in genomes:
            # This requires storing noise trajectory during generation
            log_prob = self.hypernetwork.compute_log_prob(genome)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)

        # REINFORCE gradient
        advantages = fitness - self.baseline_fitness
        loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update baseline
        self.baseline_fitness = 0.99 * self.baseline_fitness + 0.01 * fitness.mean()
```

**Note**: This requires the diffusion model to track the noise trajectory during generation so we can compute log p_θ(weights|genome).

---

## Integration with Existing Training Infrastructure

### 7. New Training Loop (`metta/rl/training/core_replicator.py`)

Replace `CoreTrainingLoop` with `ReplicatorTrainingLoop`:

```python
class ReplicatorTrainingLoop:
    """
    Training loop for replicator dynamics

    Replaces gradient-based updates with evolutionary dynamics
    """

    def __init__(self,
                 diffusion_hypernetwork: DiffusionHypernetwork,
                 population_manager: PopulationManager,
                 latent_distribution: LatentDistribution,
                 fitness_estimator: FitnessEstimator,
                 hypernetwork_trainer: HypernetworkTrainer,
                 device: torch.device,
                 context: ComponentContext):
        ...

    def run_generation(self, env: TrainingEnvironment, context: ComponentContext):
        """
        One generation = one population evaluation cycle

        Analogous to one epoch in PPO training

        Steps:
        1. Sample population from p(x)
        2. Generate policy weights for each genome via diffusion
        3. Evaluate fitness across vec_envs (Option A strategy)
        4. Update fitness estimator with new observations
        5. Update latent distribution via replicator equation
        6. Update diffusion model parameters via REINFORCE
        7. Log statistics
        """

        # 1. Sample population
        population = self.population_manager.sample_population()
        genomes, policies = zip(*population)

        # 2. Evaluate fitness
        # Option A: Each genome gets subset of vec_envs
        fitness_values = self._evaluate_population(policies, env)

        # 3. Update fitness estimator (Bayesian model)
        for genome, fitness in zip(genomes, fitness_values):
            self.fitness_estimator.update(genome, fitness)

        # 4. Update latent distribution (replicator dynamics)
        self.latent_distribution.update(
            genomes=torch.stack(genomes),
            fitness=torch.tensor(fitness_values),
            dt=0.1
        )

        # 5. Update hypernetwork (diffusion model training)
        self.hypernetwork_trainer.update(
            genomes=torch.stack(genomes),
            fitness=torch.tensor(fitness_values)
        )

        # 6. Log statistics
        stats = {
            'fitness_mean': fitness_values.mean(),
            'fitness_std': fitness_values.std(),
            'fitness_max': fitness_values.max(),
            'distribution_entropy': self.latent_distribution.entropy(),
            'hypernetwork_loss': self.hypernetwork_trainer.last_loss
        }

        return stats

    def _evaluate_population(self,
                            policies: List[Policy],
                            env: TrainingEnvironment) -> List[float]:
        """
        Option A implementation:
        - Split vec_envs among population
        - Each policy evaluated on subset
        - Average episode rewards
        """
        population_size = len(policies)
        total_envs = env.batch_info.num_envs
        envs_per_policy = total_envs // population_size

        fitness_list = []

        for policy_idx, policy in enumerate(policies):
            # Get env slice for this policy
            env_start = policy_idx * envs_per_policy
            env_end = (policy_idx + 1) * envs_per_policy
            env_slice = slice(env_start, env_end)

            # Run episode
            episode_rewards = []
            obs, _, _, _, infos, env_id_slice, _, _ = env.get_observations()

            dones = torch.zeros(envs_per_policy, dtype=torch.bool)
            total_rewards = torch.zeros(envs_per_policy)

            while not dones.all():
                # Forward pass for this policy's envs only
                obs_subset = obs[env_slice]
                actions = policy(obs_subset)

                # Step environment
                env.send_actions(actions.cpu().numpy())
                obs, _, _, _, infos, _, _, _ = env.get_observations()

                # Collect rewards
                for info in infos:
                    if 'reward' in info:
                        agent_env_idx = info['env_idx']
                        if env_start <= agent_env_idx < env_end:
                            local_idx = agent_env_idx - env_start
                            total_rewards[local_idx] += info['reward']

                    if 'episode_end' in info:
                        agent_env_idx = info['env_idx']
                        if env_start <= agent_env_idx < env_end:
                            local_idx = agent_env_idx - env_start
                            dones[local_idx] = True

            # Average fitness across this policy's envs
            fitness = total_rewards.mean().item()
            fitness_list.append(fitness)

        return fitness_list
```

---

### 8. Modified Trainer (`metta/rl/trainer_replicator.py`)

Create new trainer class that uses replicator dynamics:

```python
class ReplicatorTrainer:
    """
    Trainer for replicator dynamics

    Similar interface to standard Trainer but uses evolutionary updates
    """

    def __init__(self,
                 cfg: ReplicatorTrainerConfig,
                 env: TrainingEnvironment,
                 diffusion_hypernetwork: DiffusionHypernetwork,
                 device: torch.device,
                 distributed_helper: Optional[DistributedHelper] = None,
                 run_name: Optional[str] = None):

        self.cfg = cfg
        self.env = env
        self.device = device

        # Initialize replicator components
        self.latent_distribution = LatentDistribution(genome_dim=16)
        self.fitness_estimator = FitnessEstimator(genome_dim=16)
        self.population_manager = PopulationManager(
            population_size=cfg.population_size,
            diffusion_hypernetwork=diffusion_hypernetwork,
            latent_distribution=self.latent_distribution
        )
        self.hypernetwork_trainer = HypernetworkTrainer(diffusion_hypernetwork)

        # Training loop
        self.training_loop = ReplicatorTrainingLoop(
            diffusion_hypernetwork=diffusion_hypernetwork,
            population_manager=self.population_manager,
            latent_distribution=self.latent_distribution,
            fitness_estimator=self.fitness_estimator,
            hypernetwork_trainer=self.hypernetwork_trainer,
            device=device,
            context=self._build_context()
        )

        # Reuse existing infrastructure
        self.distributed_helper = distributed_helper or DistributedHelper(cfg.system)

    def train(self) -> None:
        """Main training loop - generations instead of epochs"""

        total_generations = self.cfg.total_evaluations // self.cfg.population_size

        for generation in range(total_generations):
            # Run one generation (replaces one PPO epoch)
            stats = self.training_loop.run_generation(self.env, self.context)

            # Update context
            self.context.generation = generation
            self.context.total_evaluations += self.cfg.population_size

            # Invoke callbacks (reuse infrastructure)
            self._invoke_callback(TrainerCallback.EPOCH_END, generation)

            if generation % self.cfg.eval_interval == 0:
                self._run_evaluation(generation)
```

---

### 9. New Config (`metta/rl/replicator/replicator_config.py`)

```python
class DiffusionHypernetworkConfig(Config):
    """Config for genome-conditioned diffusion model"""
    genome_dim: int = 16
    T_steps: int = 10  # Diffusion steps
    hidden_dim: int = 512
    num_layers: int = 6

    # Architecture choice
    backbone: Literal["transformer", "unet", "mlp"] = "transformer"

    # Training
    learning_rate: float = 1e-4


class LatentDistributionConfig(Config):
    """Config for p(x) representation"""
    genome_dim: int = 16
    representation: Literal["gmm", "flow", "particles"] = "gmm"

    # For GMM
    num_components: int = 5

    # Replicator dynamics
    diffusion_coeff: float = 0.01
    dt: float = 0.1


class FitnessEstimatorConfig(Config):
    """Config for F(x) Bayesian model"""
    method: Literal["gp", "neural", "local_linear"] = "neural"

    # For neural surrogate
    ensemble_size: int = 5
    hidden_dim: int = 128
    history_size: int = 1000


class ReplicatorTrainerConfig(Config):
    """Main config for replicator dynamics training"""

    # Population
    population_size: int = 100
    total_evaluations: int = 100_000  # Replaces total_timesteps

    # Components
    diffusion_hypernetwork: DiffusionHypernetworkConfig = Field(default_factory=DiffusionHypernetworkConfig)
    latent_distribution: LatentDistributionConfig = Field(default_factory=LatentDistributionConfig)
    fitness_estimator: FitnessEstimatorConfig = Field(default_factory=FitnessEstimatorConfig)

    # Evaluation
    eval_interval: int = 10  # Generations between evals

    # System
    system: SystemConfig = Field(default_factory=SystemConfig)
```

---

### 10. New Tool Entry Point (`metta/tools/train_replicator.py`)

```python
class TrainReplicatorTool(Tool):
    """
    Tool for training via replicator dynamics

    Similar interface to TrainTool but uses evolutionary approach
    """

    run: Optional[str] = None

    trainer: ReplicatorTrainerConfig = Field(default_factory=ReplicatorTrainerConfig)
    training_env: TrainingEnvironmentConfig  # Reuse existing
    policy_architecture: PolicyArchitecture = Field(default_factory=ViTDefaultConfig)

    # Keep infrastructure
    stats_server_uri: Optional[str] = auto_stats_server_uri()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    group: Optional[str] = None
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)

    def invoke(self, args: dict[str, str]) -> int | None:
        """Main entry point"""

        # Setup (similar to train.py)
        seed_everything(self.system)
        device = torch.device(self.system.device)
        distributed_helper = DistributedHelper(self.system)

        # Create training environment
        env = VectorizedTrainingEnvironment(self.training_env)

        # Create diffusion hypernetwork (instead of policy)
        diffusion_model = self._create_diffusion_hypernetwork(
            env.meta_data,
            self.policy_architecture
        )

        # Create trainer
        trainer = ReplicatorTrainer(
            cfg=self.trainer,
            env=env,
            diffusion_hypernetwork=diffusion_model,
            device=device,
            distributed_helper=distributed_helper,
            run_name=self.run
        )

        # Setup WandB (reuse)
        if self.wandb.enabled:
            wandb_manager = self._build_wandb_manager(distributed_helper)
            wandb_run = wandb_manager.setup()
            trainer.register_wandb(wandb_run)

        # Train
        trainer.train()

        return 0

    def _create_diffusion_hypernetwork(self,
                                       env_metadata: EnvironmentMetaData,
                                       policy_arch: PolicyArchitecture) -> DiffusionHypernetwork:
        """
        Create diffusion model that generates policy weights

        The output structure must match ViT-LSTM architecture
        """
        # Get target policy structure
        dummy_policy = policy_arch.make(env_metadata, device='cpu')
        target_shapes = {
            name: param.shape
            for name, param in dummy_policy.named_parameters()
        }

        # Create hypernetwork
        return DiffusionHypernetwork(
            config=self.trainer.diffusion_hypernetwork,
            target_param_shapes=target_shapes,
            device=self.system.device
        )
```

---

## File Structure

```
metta/rl/replicator/
├── __init__.py
├── diffusion_hypernetwork.py     # Genome -> policy weights generator
├── latent_distribution.py         # p(x) representation and updates
├── fitness_estimator.py           # Bayesian F(x) model
├── population_manager.py          # Population sampling and eval
├── replicator_equation.py         # Distribution update rules
├── hypernetwork_trainer.py        # Training the diffusion model
└── replicator_config.py           # All configs

metta/rl/
├── trainer_replicator.py          # ReplicatorTrainer (parallel to trainer.py)

metta/rl/training/
├── core_replicator.py             # ReplicatorTrainingLoop (parallel to core.py)

metta/tools/
├── train_replicator.py            # Entry point (parallel to train.py)

experiments/recipes/
├── replicator/                    # New recipe directory
    ├── __init__.py
    ├── simple_replicator.py       # Basic replicator training recipe
    └── curriculum_replicator.py   # With curriculum learning
```

---

## Implementation Phases

### Phase 1: Core Replicator Components (Week 1-2)
**Goal**: Get basic replicator dynamics working without diffusion model training

1. ✅ Implement `LatentDistribution` with GMM representation
2. ✅ Implement `FitnessEstimator` with neural surrogate
3. ✅ Implement `replicator_equation.py` with Euler updates
4. ✅ Basic `PopulationManager` (random weights, no diffusion yet)
5. ✅ Test: Can we evolve random-initialized policies?

**Milestone**: Successfully evolve a population in latent space with random weight generation

---

### Phase 2: Diffusion Hypernetwork (Week 3-4)
**Goal**: Implement genome-conditioned diffusion model

1. ✅ Design diffusion model architecture
   - Transformer backbone
   - Genome conditioning mechanism
   - Multi-step denoising (T=10)
2. ✅ Implement weight generation
   - Generate all ViT-LSTM parameters
   - Proper initialization
3. ✅ Implement LSTM state generation
4. ✅ Test: Can we generate diverse, valid policy weights?

**Milestone**: Generate 100 different policies from different genomes, verify they work

---

### Phase 3: Hypernetwork Training (Week 5-6)
**Goal**: Co-evolve hypernetwork parameters with population

1. ✅ Implement REINFORCE-style updates
2. ✅ Track noise trajectories for log-prob computation
3. ✅ Baseline and variance reduction
4. ✅ Integrate with replicator loop
5. ✅ Test: Does hypernetwork improve with evolution?

**Milestone**: Demonstrate hypernetwork learns to produce better policies over generations

---

### Phase 4: Integration & Infrastructure (Week 7-8)
**Goal**: Full integration with existing training infrastructure

1. ✅ `ReplicatorTrainer` class
2. ✅ `TrainReplicatorTool` entry point
3. ✅ WandB logging integration
4. ✅ Evaluation infrastructure (reuse existing)
5. ✅ Curriculum learning integration
6. ✅ Recipe creation

**Milestone**: Run full training loop with all monitoring and evaluation

---

### Phase 5: Optimization & Tuning (Week 9-10)
**Goal**: Make it actually work well

1. ✅ Hyperparameter tuning
   - Population size
   - Diffusion coefficient
   - Learning rates
   - dt step size
2. ✅ Stability improvements
3. ✅ Performance optimization
4. ✅ Comparison with PPO baseline

**Milestone**: Match or exceed PPO performance on at least one task

---

## Open Questions & Decisions Needed

### Critical (Must Decide Before Phase 1)

1. **Latent Distribution Representation**
   - **Recommendation**: Gaussian Mixture Model (K=5)
   - **Alternative**: Particle-based if GMM is too restrictive
   - **Decision**: ❓ Choose GMM or Particles?

2. **Fitness Estimator**
   - **Recommendation**: Ensemble of 5 small MLPs
   - **Alternative**: GP with inducing points
   - **Decision**: ❓ Neural ensemble or GP?

3. **Hypernetwork Training Strategy**
   - **Recommendation**: REINFORCE with baseline
   - **Alternative**: ES-style perturbation gradients
   - **Decision**: ❓ REINFORCE or ES?

### Important (Decide During Phase 2)

4. **Diffusion Model Architecture**
   - Transformer vs U-Net backbone?
   - How to structure weight generation (per-layer sequential vs all-at-once)?
   - **Decision**: ❓ Architecture details?

5. **LSTM State Generation**
   - Separate network or part of diffusion output?
   - Deterministic or stochastic?
   - **Decision**: ❓ How to generate h, c?

6. **Evaluation Strategy Details**
   - How many envs per genome in Option A?
   - Single episode or multiple episodes per evaluation?
   - **Decision**: ❓ Evaluation protocol?

### Nice-to-Have (Decide During Phase 5)

7. **Checkpointing Strategy**
   - What to save: latent distribution + hypernetwork + fitness estimator?
   - How to resume training?
   - **Decision**: ❓ Implement checkpointing eventually?

8. **Multi-Objective Evolution**
   - Could evolve for fitness + diversity + robustness?
   - **Decision**: ❓ Add multi-objective support later?

---

## Key Risks & Mitigations

### Risk 1: Hypernetwork Training Instability
**Problem**: Training via REINFORCE on noisy fitness signals is notoriously unstable

**Mitigations**:
- Strong baseline (exponential moving average)
- Clip gradient norms aggressively
- Low learning rate (1e-4 or lower)
- Consider variance reduction techniques (advantage actor-critic style)
- Monitor hypernetwork parameter norm

### Risk 2: Distribution Collapse
**Problem**: p(x) may collapse to single mode, losing diversity

**Mitigations**:
- Diffusion term in replicator equation
- Monitor entropy of latent distribution
- Add explicit diversity bonus to fitness
- Consider quality-diversity approach (MAP-Elites style)

### Risk 3: Poor Sample Efficiency
**Problem**: Need many fitness evaluations to train hypernetwork

**Mitigations**:
- Start with large population (100)
- Use fitness estimator to reduce noise
- Batch updates efficiently
- Consider hybrid approach (some gradient-based updates on top)

### Risk 4: Diffusion Model Capacity
**Problem**: Generating millions of parameters from 16D may be underconstrained

**Mitigations**:
- Add reconstruction loss (generate weights, run short eval, compare to target behavior)
- Add regularization (weight sparsity, structured generation)
- Consider hierarchical generation (generate layer-by-layer)
- Increase genome dimension if needed (16→32)

### Risk 5: Credit Assignment
**Problem**: Hard to attribute fitness to genome vs hypernetwork quality

**Mitigations**:
- Track both signals separately in logging
- Use control variates (baseline policies)
- A/B test hypernetwork updates vs no updates

---

## Success Metrics

### Phase 1-2 Metrics:
- [ ] Population successfully evolves (fitness increases over generations)
- [ ] Latent distribution doesn't collapse (entropy > threshold)
- [ ] Fitness estimator achieves R² > 0.7 on held-out genomes

### Phase 3-4 Metrics:
- [ ] Hypernetwork loss decreases over training
- [ ] Policies generated from hypernetwork improve over random init
- [ ] Full pipeline runs without crashes for 1000 generations

### Phase 5 Metrics (Success Criteria):
- [ ] Achieve > 0.5x PPO performance within 10x evaluations
- [ ] Discover diverse strategies (multiple modes in latent space)
- [ ] Hypernetwork generalizes (new genomes produce reasonable policies)
- [ ] System is stable (runs for 100k evaluations without intervention)

---

## Comparison to Existing Methods

This approach is similar to but distinct from:

1. **Evolution Strategies (ES)**:
   - Similar: Evolutionary optimization without gradients
   - Different: We use replicator dynamics with continuous distribution, not discrete population

2. **POET / PAIRED**:
   - Similar: Co-evolution of policies and environment
   - Different: We're co-evolving policy generator and genome distribution

3. **HyperNEAT**:
   - Similar: Use hypernetwork to generate policy weights
   - Different: We use modern diffusion models, not CPPN

4. **Quality-Diversity (MAP-Elites)**:
   - Similar: Maintain distribution over diverse policies
   - Different: Continuous distribution in latent space, not discrete archive

---

## Next Steps

1. **Before Implementation**:
   - ❓ Make decisions on critical questions (see Open Questions section)
   - Review this plan with team
   - Set up development branch

2. **Phase 1 Start**:
   - Create `metta/rl/replicator/` directory
   - Implement `LatentDistribution` (GMM or Particles)
   - Write unit tests
   - Create simple toy problem to test replicator dynamics

3. **Tracking**:
   - Create GitHub project board for phases
   - Weekly sync meetings to review progress
   - Document learnings in separate REPLICATOR_NOTES.md

---

## References for Implementation

- **Diffusion Models**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **REINFORCE**: Simple statistical gradient-following algorithms (Williams, 1992)
- **Replicator Dynamics**: The Logic of Animal Conflict (Maynard Smith, 1973)
- **HyperNetworks**: HyperNetworks (Ha et al., 2016)
- **Evolution Strategies**: Evolution Strategies as a Scalable Alternative to RL (Salimans et al., 2017)



