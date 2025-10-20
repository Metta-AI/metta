# Replicator Dynamics - Experimental Validation Roadmap

## Overview

Before implementing replicator dynamics in the full Metta codebase, we should validate core concepts in simpler, controlled experiments. This document outlines a series of standalone experiments to:

1. Test fundamental assumptions
2. Debug individual components in isolation
3. Tune hyperparameters in fast iteration environments
4. Build intuition for the approach
5. Identify and fix issues early

**Key Change**: Use **PPO instead of REINFORCE** for training the hypernetwork, as it provides better stability and sample efficiency.

---

## Experimental Environments

We'll use progressively complex environments:

### Tier 1: Toy Problems (Minutes to train)
- **Quadratic Bowl**: Simple optimization landscape, closed-form optimal
- **Rastrigin Function**: Multi-modal, tests exploration
- **Branin Function**: 2D, well-studied for Bayesian optimization

### Tier 2: Simple Control (Hours to train)
- **CartPole**: Classic RL, discrete actions, ~20k params
- **Pendulum**: Continuous control, ~5k params
- **LunarLander**: Sparse rewards, ~10k params

### Tier 3: Mini-Environments (Days to train)
- **MiniGrid**: Simple 2D navigation, ~50k params
- **ProcGen (Easy)**: Single level, ~500k params
- **Custom ArenaLite**: Simplified Metta environment, ~1M params

---

## Experiment Series

---

## Series 1: Replicator Dynamics Fundamentals
**Goal**: Validate that replicator dynamics can optimize simple fitness landscapes

**Environment**: Quadratic bowl and Rastrigin functions (no RL, just optimization)

---

### Experiment 1.1: Basic Replicator Dynamics
**Hypothesis**: Replicator equation can optimize a simple fitness landscape

**Setup**:
```python
# Fitness landscape: f(x) = -||x||^2 (quadratic bowl)
# Genome: x in R^4
# No hypernetwork - direct optimization in genome space
```

**Implementation**:
1. Initialize GMM with K=3 components centered randomly
2. Sample N=100 genomes per generation
3. Evaluate fitness: f(x) = -||x||^2
4. Update GMM via replicator equation
5. Monitor: mean fitness, distribution entropy, distance to optimum

**Success Criteria**:
- [ ] Mean fitness increases monotonically
- [ ] Final solution within 1% of optimum
- [ ] Distribution maintains entropy > 0.5 initially, then collapses near optimum

**Ablations**:
- Effect of diffusion coefficient (D = 0, 0.01, 0.1)
- Effect of population size (N = 20, 50, 100, 200)
- Effect of dt step size (0.01, 0.1, 0.5)

**Deliverable**: Notebook with plots and hyperparameter recommendations

---

### Experiment 1.2: Multi-Modal Fitness Landscape
**Hypothesis**: Diffusion term prevents premature convergence to local optima

**Setup**:
```python
# Rastrigin function: many local optima
# f(x) = -A*n - sum(x_i^2 - A*cos(2*pi*x_i))
# Genome: x in R^6
```

**Implementation**:
1. Same as 1.1 but with Rastrigin function
2. Initialize GMM far from global optimum
3. Track which local optima the distribution finds

**Success Criteria**:
- [ ] With D=0: gets stuck in local optimum
- [ ] With D>0: explores multiple modes
- [ ] Eventually finds global optimum with sufficient D

**Key Learning**: Tune diffusion coefficient for exploration/exploitation tradeoff

**Deliverable**: Comparison plots showing exploration vs exploitation

---

### Experiment 1.3: Distribution Representation Comparison
**Hypothesis**: GMM is sufficient, but validate against alternatives

**Setup**:
- Same fitness landscapes as 1.1 and 1.2
- Compare 3 representations:
  - GMM (K=5)
  - Particle-based with KDE
  - Normalizing Flow

**Metrics**:
- Final fitness achieved
- Computational cost per generation
- Ease of implementation
- Numerical stability

**Success Criteria**:
- [ ] All three converge to similar final fitness
- [ ] Identify fastest and most stable option

**Deliverable**: Table comparing methods, recommendation for main implementation

---

## Series 2: Fitness Estimation
**Goal**: Validate Bayesian fitness surrogate reduces evaluation noise

**Environment**: Noisy quadratic bowl

---

### Experiment 2.1: Neural Fitness Surrogate
**Hypothesis**: Ensemble of MLPs can accurately model fitness landscape

**Setup**:
```python
# Noisy fitness: f(x) = -||x||^2 + N(0, sigma=0.5)
# Genome: x in R^8
# Fitness estimator: Ensemble of 5 MLPs
```

**Implementation**:
1. Collect 500 (genome, noisy_fitness) pairs
2. Train ensemble of MLPs to predict fitness
3. Evaluate on held-out genomes
4. Use ensemble disagreement as uncertainty estimate

**Metrics**:
- R² on test set
- Calibration of uncertainty estimates
- Benefit in replicator updates (with vs without surrogate)

**Success Criteria**:
- [ ] R² > 0.8 on test set
- [ ] Using surrogate converges 2x faster than raw noisy fitness
- [ ] Uncertainty estimates are calibrated (disagreement ~ actual error)

**Deliverable**: Trained ensemble model and evaluation metrics

---

### Experiment 2.2: Compare GP vs Neural Surrogate
**Hypothesis**: Neural ensemble is more scalable than GP

**Setup**:
- Same as 2.1
- Compare:
  - Neural ensemble
  - Sparse GP (100 inducing points)
  - Exact GP (will be slow)

**Metrics**:
- Prediction accuracy
- Training time
- Inference time per genome
- Memory usage

**Success Criteria**:
- [ ] Neural ensemble is 10x faster than GP for N>500 samples
- [ ] Similar prediction accuracy

**Deliverable**: Benchmark results and recommendation

---

## Series 3: Hypernetwork Weight Generation
**Goal**: Test diffusion model's ability to generate valid neural network weights

**Environment**: CartPole (simple, fast)

---

### Experiment 3.1: Diffusion Model Sanity Check
**Hypothesis**: Diffusion model can generate diverse, valid network weights

**Setup**:
```python
# Target: Small MLP for CartPole (2 layers, ~1k params)
# Genome: x in R^8
# Diffusion: 10 steps, simple transformer backbone
# No training yet - just test generation
```

**Implementation**:
1. Initialize diffusion model randomly
2. Generate 100 different weight sets from random genomes
3. Test each on CartPole for 10 episodes
4. Measure diversity and basic functionality

**Metrics**:
- Weight diversity (pairwise L2 distance)
- Functionality: do all networks at least survive > 10 steps?
- Distribution of performance

**Success Criteria**:
- [ ] All generated networks are valid (no NaN, inf)
- [ ] High diversity (mean pairwise distance > threshold)
- [ ] At least 20% of networks get reward > 20 (better than random)

**Deliverable**: Analysis of generated weight distributions

---

### Experiment 3.2: Genome → Weights Mapping Quality
**Hypothesis**: Small changes in genome should lead to continuous changes in behavior

**Setup**:
```python
# Same as 3.1
# Test genome interpolation
```

**Implementation**:
1. Sample two genomes x1, x2
2. Generate weights w1 = diffusion(x1), w2 = diffusion(x2)
3. Interpolate: w_interp = diffusion((1-t)*x1 + t*x2) for t in [0, 1]
4. Measure behavior continuity

**Metrics**:
- Smoothness of policy behavior along interpolation path
- Correlation between genome distance and weight distance
- Correlation between genome distance and behavior distance

**Success Criteria**:
- [ ] Interpolated policies show smooth behavioral transition
- [ ] Genome distance correlates with behavior distance (r > 0.6)

**Deliverable**: Visualization of genome space and behavior space

---

### Experiment 3.3: Diffusion Model Architecture Comparison
**Hypothesis**: Find best architecture for weight generation

**Setup**:
- Target: CartPole MLP (~1k params)
- Compare architectures:
  - Simple MLP decoder
  - Transformer backbone
  - U-Net style (if applicable)
  - Hierarchical (layer-by-layer generation)

**Metrics**:
- Generation quality (diversity + functionality)
- Training stability (in next series)
- Memory usage
- Inference speed

**Success Criteria**:
- [ ] Identify architecture with best quality/speed tradeoff

**Deliverable**: Architecture recommendation for main implementation

---

## Series 4: Hypernetwork Training with PPO
**Goal**: Train the diffusion model using PPO to generate better policies

**Environment**: CartPole → Pendulum

---

### Experiment 4.1: PPO for Hypernetwork Training
**Hypothesis**: PPO can train hypernetwork parameters using fitness as reward

**Setup**:
```python
# Target: CartPole MLP (~1k params)
# Hypernetwork: Diffusion model with ~100k params
# Training: PPO on hypernetwork parameters
# "Episode": Sample genome → Generate weights → Evaluate fitness
```

**Key Insight**: Treat hypernetwork parameter updates as a meta-RL problem
- **State**: Current hypernetwork parameters (or latent state)
- **Action**: Parameter update direction
- **Reward**: Average fitness of generated policies

**Implementation**:

1. **Diffusion Model as Policy**:
   ```python
   # The diffusion model is parameterized by θ
   # It defines a stochastic mapping: genome -> weights

   def generate_policy_weights(genome, diffusion_params_θ, noise_ε):
       # Stochastic generation allows for policy gradient
       weights = diffusion_model(genome, noise_ε; θ)
       return weights
   ```

2. **PPO Training Loop**:
   ```python
   for iteration in range(num_iterations):
       # Rollout: Sample genomes and generate policies
       genomes = sample_from_latent_distribution(N=50)

       trajectories = []
       for genome in genomes:
           # Generate weights stochastically
           noise = sample_noise()
           weights = diffusion_model(genome, noise; θ)

           # Evaluate in CartPole
           fitness = evaluate_policy(weights, num_episodes=3)

           # Store trajectory for PPO
           trajectories.append({
               'genome': genome,
               'noise': noise,
               'fitness': fitness,
               'log_prob': compute_log_prob(noise, genome; θ)
           })

       # PPO update on hypernetwork parameters θ
       ppo_update(θ, trajectories)
   ```

3. **Value Function**:
   ```python
   # Value network estimates expected fitness for a genome
   V(genome) = expected_fitness_under_current_hypernetwork

   # This serves as baseline for variance reduction
   advantage = fitness - V(genome)
   ```

4. **PPO Objective**:
   ```python
   # Standard PPO clipped objective
   ratio = π_θ_new(noise|genome) / π_θ_old(noise|genome)

   L_CLIP = min(
       ratio * advantage,
       clip(ratio, 1-ε, 1+ε) * advantage
   )

   L_VF = (fitness - V(genome))^2

   L_total = L_CLIP - c1 * L_VF + c2 * entropy_bonus
   ```

**Success Criteria**:
- [ ] Hypernetwork improves: avg fitness increases over iterations
- [ ] Training is stable (no catastrophic forgetting)
- [ ] Generated policies outperform random hypernetwork after 100 updates
- [ ] Converges to CartPole solution (reward ~500) within 1000 iterations

**Ablations**:
- PPO clip coefficient (ε = 0.1, 0.2, 0.3)
- Number of evaluation episodes per genome (1, 3, 5)
- Batch size (N = 20, 50, 100)
- Value function architecture

**Deliverable**:
- Trained hypernetwork that generates good CartPole policies
- Training curves and hyperparameter recommendations
- Comparison to REINFORCE baseline

---

### Experiment 4.2: PPO + Replicator Dynamics Co-Evolution
**Hypothesis**: Combining PPO hypernetwork training with replicator dynamics distribution updates works better than either alone

**Setup**:
```python
# CartPole MLP
# Train both:
#   1. Hypernetwork parameters (via PPO)
#   2. Latent distribution (via replicator equation)
```

**Implementation**:
```python
for generation in range(num_generations):
    # Sample from current latent distribution
    genomes = latent_distribution.sample(N=50)

    # Generate policies and evaluate
    fitness_values = []
    ppo_trajectories = []
    for genome in genomes:
        noise = sample_noise()
        weights = diffusion_model(genome, noise; θ)
        fitness = evaluate(weights)

        fitness_values.append(fitness)
        ppo_trajectories.append({
            'genome': genome,
            'noise': noise,
            'fitness': fitness,
            'log_prob': compute_log_prob(noise, genome; θ)
        })

    # Update 1: PPO on hypernetwork
    ppo_update(θ, ppo_trajectories)

    # Update 2: Replicator dynamics on distribution
    latent_distribution.update(genomes, fitness_values, dt=0.1)
```

**Key Question**: Do these two updates interfere or synergize?

**Success Criteria**:
- [ ] Co-evolution converges faster than either alone
- [ ] Distribution evolves toward high-fitness regions
- [ ] Hypernetwork learns to generate better policies
- [ ] Final performance > baseline methods

**Metrics to Track**:
- Hypernetwork loss over time
- Distribution entropy over time
- Mean fitness over time
- Best fitness found so far
- Diversity of policies in population

**Deliverable**:
- Analysis of interaction between two update mechanisms
- Recommendations for update frequency ratio (e.g., 1 PPO : 10 replicator updates)

---

### Experiment 4.3: Scale to Pendulum
**Hypothesis**: Approach scales to continuous control

**Setup**:
- Pendulum environment (continuous actions)
- Larger policy network (~5k params)
- Same PPO + replicator dynamics setup

**Changes from CartPole**:
- Continuous action space
- Longer episodes
- Different reward scale

**Success Criteria**:
- [ ] Achieves near-optimal Pendulum performance (reward > -200)
- [ ] Training remains stable
- [ ] Computational cost is reasonable

**Deliverable**: Validation that approach works for continuous control

---

## Series 5: LSTM State Handling
**Goal**: Test LSTM hidden state generation and evolution

**Environment**: Memory-based CartPole (obscured state)

---

### Experiment 5.1: LSTM State Generation
**Hypothesis**: Diffusion model can generate functional LSTM initial states

**Setup**:
```python
# CartPole with partial observability (velocity hidden)
# Policy: LSTM (requires memory)
# Genome: x in R^16
# Generate: weights + initial (h0, c0)
```

**Implementation**:
1. Extend diffusion model to output (weights, h0, c0)
2. Sample genomes and generate LSTM policies
3. Test on memory-dependent task
4. Compare to fixed h0=0 initialization

**Success Criteria**:
- [ ] Generated h0, c0 improve performance over zero init
- [ ] Different genomes produce different memory strategies
- [ ] LSTM successfully learns to use memory

**Deliverable**: Evidence that LSTM state generation is beneficial

---

### Experiment 5.2: Evolution of Memory Strategies
**Hypothesis**: Replicator dynamics can evolve different memory strategies

**Setup**:
- Same as 5.1
- Run full co-evolution (PPO + replicator)
- Analyze evolved memory usage patterns

**Analysis**:
- Cluster genomes by behavior
- Visualize LSTM hidden state trajectories
- Identify distinct memory strategies

**Success Criteria**:
- [ ] Multiple distinct memory strategies emerge
- [ ] Different strategies perform well in different scenarios

**Deliverable**: Characterization of evolved memory diversity

---

## Series 6: Scalability Testing
**Goal**: Test approach on larger networks and harder tasks

**Environment**: MiniGrid → ProcGen Easy

---

### Experiment 6.1: MiniGrid Navigation
**Hypothesis**: Approach works on simple grid world with ~50k params

**Setup**:
- MiniGrid-Empty-8x8
- CNN policy (~50k params)
- Full pipeline: diffusion + PPO + replicator

**Success Criteria**:
- [ ] Solves MiniGrid (success rate > 90%)
- [ ] Training time < 12 hours on single GPU
- [ ] Memory usage acceptable

**Key Challenges**:
- Larger weight generation
- More complex environment
- Longer episodes

**Deliverable**: First validation on vision-based task

---

### Experiment 6.2: ProcGen Easy Level
**Hypothesis**: Approach scales to procedurally generated environments

**Setup**:
- ProcGen-CoinRun (easy mode, single level)
- CNN policy (~500k params)
- May need to increase genome dimension (16 → 32)

**Success Criteria**:
- [ ] Achieves reasonable performance (> random, < PPO baseline)
- [ ] Demonstrates generalization within level
- [ ] Identifies whether approach is competitive

**Key Risk**: This may be too hard - if so, return to MiniGrid with curriculum

**Deliverable**: Assessment of approach's viability for complex RL

---

## Series 7: Integration Testing
**Goal**: Test components work together before main implementation

**Environment**: Custom ArenaLite (simplified Metta)

---

### Experiment 7.1: ArenaLite Prototype
**Hypothesis**: Full system works on Metta-like environment

**Setup**:
- Simplified Metta environment:
  - 16x16 grid
  - 10 agents
  - Basic combat mechanics
  - ~1M params (small ViT-LSTM)
- Full pipeline with all components

**Implementation**:
1. Create ArenaLite environment (weekend project)
2. Set up complete pipeline
3. Train for 1M environment steps
4. Compare to PPO baseline

**Success Criteria**:
- [ ] All components integrate successfully
- [ ] Training runs without crashes
- [ ] Performance is reasonable (>50% of PPO)
- [ ] Identifies any integration issues

**Key Learnings**:
- Performance bottlenecks
- Memory issues
- Stability problems
- Needed adjustments before main implementation

**Deliverable**:
- Complete prototype system
- List of issues to fix
- Performance baseline
- Confidence for main implementation

---

## Experiment Infrastructure

### Code Organization
```
replicator_experiments/
├── README.md
├── requirements.txt
├── common/
│   ├── __init__.py
│   ├── latent_distribution.py      # Shared across experiments
│   ├── fitness_estimator.py
│   ├── diffusion_hypernetwork.py
│   ├── replicator_equation.py
│   └── ppo_trainer.py              # PPO for hypernetwork
├── series_1_replicator_basics/
│   ├── exp_1_1_quadratic.py
│   ├── exp_1_2_rastrigin.py
│   └── exp_1_3_distribution_comparison.py
├── series_2_fitness_estimation/
│   ├── exp_2_1_neural_surrogate.py
│   └── exp_2_2_gp_comparison.py
├── series_3_weight_generation/
│   ├── exp_3_1_diffusion_sanity.py
│   ├── exp_3_2_genome_mapping.py
│   └── exp_3_3_architecture_comparison.py
├── series_4_ppo_training/
│   ├── exp_4_1_ppo_hypernetwork.py
│   ├── exp_4_2_coevolution.py
│   └── exp_4_3_pendulum.py
├── series_5_lstm/
│   ├── exp_5_1_state_generation.py
│   └── exp_5_2_memory_evolution.py
├── series_6_scalability/
│   ├── exp_6_1_minigrid.py
│   └── exp_6_2_procgen.py
├── series_7_integration/
│   ├── arena_lite_env.py
│   └── exp_7_1_full_prototype.py
└── notebooks/
    ├── 01_replicator_visualization.ipynb
    ├── 02_fitness_landscapes.ipynb
    ├── 03_hypernetwork_analysis.ipynb
    └── 04_results_summary.ipynb
```

### Compute Requirements

**Series 1-2**: Laptop CPU (minutes to hours)
**Series 3-4**: Single GPU (hours)
**Series 5-6**: Single GPU (days)
**Series 7**: 2-4 GPUs (days)

### Time Estimates

| Series | Time to Complete | Parallelizable? |
|--------|------------------|-----------------|
| Series 1 | 3-5 days | Yes (experiments independent) |
| Series 2 | 2-3 days | Yes |
| Series 3 | 3-5 days | Partially |
| Series 4 | 1-2 weeks | No (sequential refinement) |
| Series 5 | 3-5 days | Partially |
| Series 6 | 1-2 weeks | Partially |
| Series 7 | 1 week | No |

**Total**: 6-10 weeks if done sequentially, 4-6 weeks with parallelization

---

## Key Differences from Main Plan: PPO for Hypernetwork

### Why PPO Instead of REINFORCE?

1. **Stability**: Clipped objective prevents destructive updates
2. **Sample Efficiency**: Value function reduces variance
3. **Proven**: PPO is more reliable than vanilla policy gradient
4. **Natural Fit**: Treating hypernetwork training as meta-RL problem

### PPO Implementation Details

**"State"**:
- Not used directly (or could be latent distribution statistics)
- We're doing episodic RL where each "episode" is a genome evaluation

**"Action"**:
- Noise vector ε sampled during diffusion generation
- Or equivalently, the stochasticity in weight generation

**"Policy"**:
- π_θ(ε|genome) = diffusion model's conditional distribution
- Parameterized by hypernetwork weights θ

**"Value Function"**:
- V(genome) = expected fitness under current hypernetwork
- Trained on same data as policy
- Predicts how good a genome will be

**"Reward"**:
- R = fitness from evaluating generated policy
- Single reward per "episode" (one genome evaluation)

**"Advantage"**:
- A(genome) = fitness - V(genome)
- Reduces variance by centering around expectation

### PPO Update Equations

```python
# Collect rollouts
for i in range(batch_size):
    genome_i = sample_genome()
    noise_i = sample_noise()
    weights_i = diffusion_model(genome_i, noise_i; θ_old)
    fitness_i = evaluate(weights_i)
    log_prob_old_i = log π_θ_old(noise_i | genome_i)
    value_i = V(genome_i)
    advantage_i = fitness_i - value_i

# PPO update
for epoch in range(K):  # K=5 typically
    log_prob_new = log π_θ_new(noise_i | genome_i)
    ratio = exp(log_prob_new - log_prob_old_i)

    # Clipped surrogate objective
    L_CLIP = min(
        ratio * advantage_i,
        clip(ratio, 1-ε, 1+ε) * advantage_i
    )

    # Value function loss
    L_VF = (V(genome_i) - fitness_i)^2

    # Total loss
    loss = -mean(L_CLIP) + c1 * mean(L_VF) - c2 * entropy(π_θ)

    # Gradient update
    θ = θ - α * ∇_θ loss
```

### Computing Log Probability

The diffusion model needs to support log probability computation:

```python
class DiffusionHypernetwork:
    def generate_and_log_prob(self, genome, noise):
        """
        Generate weights and compute log probability

        For diffusion models, this requires tracking the
        denoising trajectory and using the learned score function
        """
        weights = self.generate(genome, noise)

        # Compute log p(noise | genome)
        # This is the negative ELBO or score matching objective
        log_prob = -self.denoising_loss(genome, noise)

        return weights, log_prob
```

---

## Success Criteria for Experiments

### Tier 1: Must Achieve (Go/No-Go)
- [ ] Series 1: Replicator dynamics converges on toy problems
- [ ] Series 4.1: PPO successfully trains hypernetwork on CartPole
- [ ] Series 4.2: Co-evolution shows synergy between updates

### Tier 2: Should Achieve (Confidence Builders)
- [ ] Series 2: Fitness surrogate achieves R² > 0.8
- [ ] Series 3: Generated weights are diverse and functional
- [ ] Series 4.3: Scales to continuous control (Pendulum)

### Tier 3: Nice to Have (Validation)
- [ ] Series 5: LSTM state generation improves memory tasks
- [ ] Series 6: Works on vision-based task (MiniGrid)
- [ ] Series 7: Full prototype achieves >50% of PPO baseline

### Stopping Criteria

**Proceed to Main Implementation if**:
- All Tier 1 criteria met
- At least 2/3 Tier 2 criteria met
- No fundamental blockers identified

**Pivot/Rethink if**:
- PPO training doesn't improve hypernetwork (Series 4.1 fails)
- Co-evolution is unstable (Series 4.2 fails)
- Approach doesn't scale beyond toy problems

---

## Reporting and Documentation

### Weekly Updates
- Summary of experiments completed
- Key findings and surprises
- Updated hyperparameter recommendations
- Blockers and risks

### Final Report
- Complete results from all experiments
- Hyperparameter recommendations for main implementation
- Identified risks and mitigation strategies
- Updated implementation plan based on learnings
- Go/no-go recommendation

### Code Deliverables
- Clean, documented implementations of all components
- Reusable modules for main implementation
- Notebooks with visualizations and analysis
- Benchmark results on standard tasks

---

## Risk Mitigation Through Experiments

| Risk from Main Plan | Experiments That Address It | Mitigation Strategy |
|---------------------|------------------------------|---------------------|
| Hypernetwork training instability | Series 4 (PPO training) | Test PPO vs REINFORCE, tune clip coeff |
| Distribution collapse | Series 1.2 (multi-modal) | Test diffusion term, tune D coefficient |
| Poor sample efficiency | Series 2 (fitness surrogate), Series 4 (PPO) | Validate surrogate works, use PPO not REINFORCE |
| Diffusion capacity issues | Series 3 (weight generation) | Test architectures, increase genome dim if needed |
| Credit assignment | Series 4.2 (co-evolution) | Track signals separately, test update ratios |
| LSTM complexity | Series 5 (LSTM states) | Validate generation works before main impl |
| Doesn't scale | Series 6 (scalability) | Test progressively harder tasks, identify limits |

---

## Open Questions to Answer

### Series 1-2 Should Answer:
- [ ] What diffusion coefficient D provides best exploration/exploitation?
- [ ] Is GMM sufficient or do we need normalizing flows?
- [ ] Does fitness surrogate provide 2x speedup as hypothesized?

### Series 3-4 Should Answer:
- [ ] Which diffusion architecture is best for weight generation?
- [ ] Does PPO significantly outperform REINFORCE for hypernetwork training?
- [ ] What's the right ratio of PPO updates to replicator updates?
- [ ] Should we update every generation or accumulate across multiple?

### Series 5-6 Should Answer:
- [ ] Is LSTM state generation necessary or can we use zero init?
- [ ] At what policy size does the approach break down?
- [ ] Do we need to increase genome dimension for complex tasks?

### Series 7 Should Answer:
- [ ] Are there any show-stopping integration issues?
- [ ] What's the realistic performance gap vs PPO?
- [ ] Is the approach worth implementing in main codebase?

---

## Next Steps

1. **Week 1**: Set up experiment infrastructure
   - Create repository
   - Install dependencies (gym, stable-baselines3, etc.)
   - Implement basic components (GMM, simple diffusion model)

2. **Week 2-3**: Run Series 1-2
   - Validate replicator dynamics
   - Test fitness estimation
   - First checkpoint: Go/no-go on fundamentals

3. **Week 4-6**: Run Series 3-4
   - Test weight generation
   - Implement PPO training
   - Test co-evolution
   - Second checkpoint: Go/no-go on core approach

4. **Week 7-8**: Run Series 5-6 (if passing checkpoints)
   - LSTM states
   - Scalability tests
   - Third checkpoint: Performance assessment

5. **Week 9-10**: Series 7 + Final report
   - Full prototype
   - Compile recommendations
   - Update main implementation plan
   - Final go/no-go decision

---

## Collaboration Plan

**Solo Work**:
- Individual experiment implementation
- Data collection and analysis

**Pair Programming**:
- PPO implementation (Series 4)
- Diffusion architecture design (Series 3)
- ArenaLite environment (Series 7)

**Group Reviews**:
- Weekly: Progress and findings
- After each series: Results and implications
- Final: Go/no-go decision

---

## Appendix: PPO Pseudocode for Hypernetwork Training

```python
class PPOHypernetworkTrainer:
    def __init__(self,
                 diffusion_model: DiffusionHypernetwork,
                 value_network: ValueNetwork,
                 clip_epsilon: float = 0.2,
                 ppo_epochs: int = 5,
                 batch_size: int = 64):
        self.diffusion_model = diffusion_model
        self.value_network = value_network
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.policy_optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=3e-4
        )
        self.value_optimizer = torch.optim.Adam(
            value_network.parameters(), lr=1e-3
        )

    def collect_rollouts(self, latent_distribution, num_rollouts=64):
        """Collect rollouts by sampling genomes and evaluating policies"""
        rollouts = []

        for _ in range(num_rollouts):
            # Sample genome from current distribution
            genome = latent_distribution.sample(1)[0]

            # Generate policy stochastically
            noise = torch.randn_like(self.diffusion_model.noise_shape)
            weights, log_prob_old = self.diffusion_model.generate_and_log_prob(
                genome, noise
            )

            # Evaluate policy
            policy = self.instantiate_policy(weights)
            fitness = self.evaluate_policy(policy, num_episodes=3)

            # Predict value
            value = self.value_network(genome)

            rollouts.append({
                'genome': genome,
                'noise': noise,
                'weights': weights,
                'fitness': fitness,
                'log_prob_old': log_prob_old,
                'value': value
            })

        return rollouts

    def train_step(self, rollouts):
        """Perform PPO update on hypernetwork parameters"""

        # Compute advantages
        for rollout in rollouts:
            rollout['advantage'] = rollout['fitness'] - rollout['value'].detach()

        # Normalize advantages
        advantages = torch.tensor([r['advantage'] for r in rollouts])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Shuffle rollouts
            indices = torch.randperm(len(rollouts))

            for start_idx in range(0, len(rollouts), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch = [rollouts[i] for i in batch_indices]

                # Recompute log probs with current policy
                log_probs_new = []
                for rollout in batch:
                    _, log_prob = self.diffusion_model.generate_and_log_prob(
                        rollout['genome'], rollout['noise']
                    )
                    log_probs_new.append(log_prob)

                log_probs_new = torch.stack(log_probs_new)
                log_probs_old = torch.stack([r['log_prob_old'] for r in batch])
                batch_advantages = torch.tensor([advantages[i] for i in batch_indices])

                # Compute PPO loss
                ratio = torch.exp(log_probs_new - log_probs_old)

                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages

                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Entropy bonus
                entropy = -log_probs_new.mean()

                # Total policy loss
                total_policy_loss = policy_loss - 0.01 * entropy

                # Update policy (diffusion model)
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_model.parameters(), 0.5
                )
                self.policy_optimizer.step()

                # Update value function
                values = torch.stack([self.value_network(r['genome']) for r in batch])
                targets = torch.tensor([r['fitness'] for r in batch])
                value_loss = (values - targets).pow(2).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_ratio': ratio.mean().item()
        }
```

This PPO implementation treats the hypernetwork as a meta-policy that generates policy parameters, and trains it using standard PPO with fitness as the reward signal.



