# Experimental Design Analysis: Patterson et al. (2024)

**Based on**: ["Empirical Design in Reinforcement Learning"](https://arxiv.org/abs/2304.01315)
**Authors**: Andrew Patterson, Samuel Neumann, Martha White, Adam White
**Published**: [JMLR 2024, Volume 25, Number 183](https://jmlr.org/papers/v25/23-0183.html)
**arXiv**: [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)

---

## Overview

This document analyzes Metta's current experimental and evaluation infrastructure against best practices from Patterson et al. (2024). The paper addresses critical gaps in reinforcement learning research methodology, emphasizing that "popular algorithms are sensitive to hyper-parameter settings and implementation details" and that "common empirical practice leads to weak statistical evidence."

**Purpose**: Identify gaps in our current approach and provide actionable recommendations to improve the statistical rigor and reproducibility of Metta experiments.

**Target Audience**: Researchers, ML engineers, and contributors working on training experiments, evaluation, and analysis.

---

## Table of Contents

1. [Key Principles from Patterson et al.](#key-principles-from-patterson-et-al)
2. [Current State in Metta](#current-state-in-metta)
3. [Gap Analysis](#gap-analysis)
4. [Recommendations](#recommendations)
5. [Implementation Priorities](#implementation-priorities)
6. [References](#references)

---

## Key Principles from Patterson et al.

> **Paper Summary**: "This manuscript is both a call-to-action and a comprehensive resource for how to do good experiments in reinforcement learning. In particular, it covers: the statistical assumptions underlying common performance measures, how to characterize performance variation and stability, hypothesis testing, special considerations for comparing multiple agents, baseline and illustrative example construction, and how to deal with hyper-parameters and experimenter bias."
>
> Source: [Patterson et al. (2024), Abstract](https://arxiv.org/abs/2304.01315)

### Statistical Rigor

#### 1. Sample Size Requirements

**From the paper** ([Section on Statistical Power](https://www.brandonrohrer.com/empirical_design_rl)):
- **Minimum runs**: 30+ independent runs needed for strong statistical claims
- **Insufficient**: 5 runs provide weak evidence
- **Context matters**: More runs needed for high-variance environments or small effect sizes

**Why it matters**: Small sample sizes lead to:
- High probability of false positives
- Inability to detect real differences between algorithms
- Poor generalization of results

#### 2. Reporting Standards

**From the paper**:
> "Use sample standard deviations or tolerance intervals instead of means alone"
>
> "Report confidence intervals when focusing on mean performance estimates"
>
> "Avoid standard errors - they provide low-confidence confidence intervals"

**DO**:
- Report **sample standard deviations** to show variation
- Use **confidence intervals** (bootstrap or t-based)
- Report **percentile-based metrics**: median, IQM (Interquartile Mean)
- Show **distribution plots** (kernel density estimation)

**DON'T**:
- Report only means without uncertainty measures
- Use standard errors as confidence intervals
- Cherry-pick best runs or favorable random seeds

#### 3. Visualization Best Practices

**From the paper**:
- **Individual trajectories**: "Plot single agent performance trajectories, not just aggregates"
- **Performance vs. steps**: "Plot performance versus steps of interaction, rather than episodes"
- **Distribution analysis**: "Use kernel density estimation to understand performance distributions"

### Experimental Design

#### 4. Seed Management

**CRITICAL PRINCIPLE** from the paper:

> **"The seed is NOT a tuneable hyperparameter"**
>
> Source: [Patterson et al. (2024)](https://www.brandonrohrer.com/empirical_design_rl)

**From the paper**:
- Never select seeds that produce favorable results
- Never discard "bad" runs without strong justification
- Document all runs, including failures
- Use different seeds for different purposes (training vs. evaluation)

#### 5. Performance Metrics

**From the paper**:
- **Primary metric**: Performance versus interaction steps (not episodes)
- **Reason**: Episodes can have variable lengths, making comparison unfair
- **Secondary metrics**: Sample efficiency, wall-clock time

#### 6. Baseline Construction

**From the paper**:
> "Apply equivalent optimization effort to baseline algorithms as to novel approaches"

- Don't compare heavily-tuned algorithm to vanilla baselines
- Document hyperparameter search process for all methods
- Use same computational budget for all comparisons

### Common Pitfalls

The paper identifies **20+ common mistakes**. Key ones for Metta:

1. **Inadequate sample size**: Too few runs to support claims
2. **Untuned baselines**: Comparing optimized algorithm to default baselines
3. **Seed selection bias**: "Cherrypicking prevention: Don't select environments solely where your algorithm excels"
4. **Discarded runs**: Removing "failed" runs without justification
5. **Invalid error bars**: Using inappropriate uncertainty measures
6. **Environment overfitting**: Testing only on cherry-picked tasks
7. **Missing implementation details**: "Report comprehensive implementation details and attempt external validation"
8. **Aggressive episode cutoffs**: "Avoid aggressive cutoffs that skew results; distinguish episode cutoffs from termination signals"
9. **HARK**: "Avoid HARK (Hypothesis After Results Known)"
10. **No distribution analysis**: Ignoring whether results are skewed or multi-modal

**Reference**: [20-item error checklist from Patterson et al. (2024)](https://www.brandonrohrer.com/empirical_design_rl)

---

## Current State in Metta

### What Metta Does Well ✅

#### Seed Management
**Code**: `metta/rl/system_config.py:42`

- ✅ Seeds are randomly generated by default: `seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))`
- ✅ Seed propagation to numpy, torch, and random modules
- ✅ Rank-specific seeds for distributed training
- ✅ Deterministic mode available via `torch_deterministic` flag

```python
# system_config.py:42
seed: int = Field(default_factory=lambda: np.random.randint(0, 1000000))

# system_config.py:70-74
rank_specific_seed = (seed + rank) if seed is not None else rank
random.seed(rank_specific_seed)
np.random.seed(rank_specific_seed)
torch.manual_seed(rank_specific_seed)
```

#### Logging and Tracking
**Code**: `metta/rl/training/wandb_logger.py:26`

- ✅ WandB integration for experiment tracking
- ✅ Performance logged against agent_step (not episodes): Aligns with paper's recommendation
- ✅ Comprehensive timing metrics (train, rollout, stats time)
- ✅ Steps per second tracking

#### Evaluation Infrastructure
**Code**: `metta/rl/training/evaluator.py`

- ✅ Structured evaluation system with `EvaluatorConfig`
- ✅ Local and remote evaluation support
- ✅ Evaluation database for storing results
- ✅ Per-simulation configuration

#### Analysis Tools
**Code**: `metta/eval/analysis.py:87-90`

- ✅ Compute mean and standard deviation
- ✅ Track sample counts (recorded metrics vs. potential samples)
- ✅ SQL-based statistics database for efficient queries
- ✅ Metric filtering and selection

```python
# analysis.py:87-90
mean = stats_db.get_average_metric(m, policy_uri, filter_condition)
std = stats_db.get_std_metric(m, policy_uri, filter_condition) or 0.0
k_recorded = stats_db.count_metric_agents(pk, pv, m, filter_condition)
n_potential = stats_db.potential_samples_for_metric(pk, pv, filter_condition)
```

---

## Gap Analysis

Comparing Metta's implementation against [Patterson et al. (2024) recommendations](https://arxiv.org/abs/2304.01315):

### Critical Gaps 🔴

#### 1. No Multi-Seed Infrastructure
**Paper Recommendation**: "Minimum 30+ runs needed"
**Current State**: Users must manually launch multiple training runs with different seeds
**Impact**: Hard to achieve statistically sufficient experiments

**Location**: `metta/tools/train.py`

**What's missing**:
```bash
# This doesn't exist:
uv run ./tools/run.py train arena run=my_exp seeds=30

# Users must do:
for i in {1..30}; do
  uv run ./tools/run.py train arena run=my_exp_seed_$i system.seed=$i
done
```

#### 2. No Confidence Intervals
**Paper Recommendation**: "Report confidence intervals when focusing on mean performance estimates"
**Current State**: Analysis reports only mean and standard deviation
**Impact**: Cannot make strong claims about algorithm performance differences

**Location**: `metta/eval/analysis.py:104-124`

```python
# Current output:
headers = ["Metric", "Average", "Std Dev", "Metric Samples", "Agent Samples"]
```

**Missing** (per paper):
- Bootstrap confidence intervals
- Percentile-based metrics (median, IQM)
- Distribution shape information
- Statistical significance testing

#### 3. No Individual Trajectory Visualization
**Paper Recommendation**: "Plot single agent performance trajectories, not just aggregates"
**Current State**: Only aggregated metrics logged to WandB
**Impact**: Can't identify outlier runs or see variance over time

**Location**: `metta/rl/training/wandb_logger.py`

### Major Gaps 🟡

#### 4. No Distribution Analysis
**Paper Recommendation**: "Use kernel density estimation to understand performance distributions, not just summary statistics"
**Current State**: Only mean/std summary statistics
**Impact**: May miss important patterns (e.g., bimodal performance, heavy tails)

**Location**: `metta/eval/analysis.py`

#### 5. No Statistical Hypothesis Testing
**Paper Context**: Discusses proper hypothesis testing and multiple comparison corrections
**Current State**: No statistical tests for comparing algorithms
**Impact**: Cannot make statistically rigorous claims about which algorithm is better

**Missing**:
- Welch's t-test for comparing two algorithms
- Bootstrap hypothesis tests
- Multiple comparison corrections (e.g., Bonferroni, Holm-Bonferroni)
- Effect size reporting (Cohen's d)

#### 6. No Explicit Seed Tuning Guardrails
**Paper Warning**: **"The seed is NOT a tuneable hyperparameter"**
**Current State**: No warnings or documentation about seed selection bias
**Impact**: Easy for users to unknowingly introduce bias

**Location**: Documentation and training code

### Minor Gaps 🟢

#### 7. Limited WandB Metrics Reporting
**Paper Recommendation**: Report uncertainty alongside point estimates
**Current State**: Point estimates logged without uncertainty
**Location**: `metta/rl/training/stats_reporter.py:44-110`

#### 8. No Standardized Experiment Templates
**Paper Emphasis**: "Report comprehensive implementation details"
**Current State**: No templates for reporting experimental results
**Impact**: Inconsistent experiment documentation

---

## Recommendations

### High Priority (Implement First)

#### R1: Add Multi-Seed Training Support

**Paper Justification**: "30+ runs needed" ([Patterson et al., 2024](https://www.brandonrohrer.com/empirical_design_rl))

**Proposed Implementation**:
```python
# In metta/tools/multi_seed_train.py (new file)

class MultiSeedTrainer:
    """Run the same training configuration with multiple random seeds."""

    def __init__(self, base_config, num_seeds: int = 30):
        self.base_config = base_config
        self.num_seeds = num_seeds

    def run_all_seeds(self):
        """Launch training runs with different seeds."""
        results = []
        for seed_idx in range(self.num_seeds):
            config = copy.deepcopy(self.base_config)
            config.system.seed = random.randint(0, 1_000_000)
            config.run_name = f"{config.run_name}_seed{seed_idx:03d}"

            result = self._run_single_seed(config)
            results.append(result)

        return self._aggregate_results(results)
```

**CLI Interface**:
```bash
uv run ./tools/run.py multi-seed-train arena \
  run=my_experiment \
  seeds=30 \
  parallel=4
```

**Files to create**:
- `metta/tools/multi_seed_train.py`
- `metta/rl/multi_seed_config.py`

---

#### R2: Enhance Analysis Tool with Confidence Intervals

**Paper Justification**: "Report confidence intervals" ([Patterson et al., 2024](https://arxiv.org/abs/2304.01315))

**Proposed Changes to `metta/eval/analysis.py`**:

```python
def compute_bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000):
    """Compute bootstrap confidence interval for mean."""
    bootstrap_means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = (1 - confidence) / 2
    upper = 1 - lower
    return np.percentile(bootstrap_means, [lower * 100, upper * 100])

def get_metrics_data_enhanced(stats_db, policy_uri, metrics, sim_name=None):
    """Enhanced version with confidence intervals and percentiles."""
    # ... existing code ...

    for m in metrics:
        values = stats_db.get_metric_values(m, policy_uri, filter_condition)

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        median = np.median(values)

        # IQM - mean of middle 50%
        q25, q75 = np.percentile(values, [25, 75])
        iqm_values = values[(values >= q25) & (values <= q75)]
        iqm = np.mean(iqm_values)

        # Bootstrap 95% CI
        ci_low, ci_high = compute_bootstrap_ci(values)

        data[m] = {
            "mean": mean,
            "std": std,
            "median": median,
            "iqm": iqm,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "count": len(values),
        }
```

**New Output**:
```
┌─────────┬────────┬────────┬────────┬──────────────────┬──────────┬─────┐
│ Metric  │ Mean   │ Median │ IQM    │ 95% CI           │ Std Dev  │  N  │
├─────────┼────────┼────────┼────────┼──────────────────┼──────────┼─────┤
│ reward  │ 0.8523 │ 0.8501 │ 0.8512 │ [0.831, 0.874]   │ 0.1205   │ 30  │
└─────────┴────────┴────────┴────────┴──────────────────┴──────────┴─────┘
```

**Files to modify**:
- `metta/eval/analysis.py`
- `metta/eval/eval_stats_db.py` (add `get_metric_values()`)
- `metta/eval/analysis_config.py`

---

#### R3: Add Seed Tuning Warning to Documentation

**Paper Justification**: **"The seed is NOT a tuneable hyperparameter"** ([Patterson et al., 2024](https://www.brandonrohrer.com/empirical_design_rl))

**Add to `CLAUDE.md`**:

```markdown
### Random Seed Management

#### CRITICAL: Seeds Are NOT Hyperparameters

⚠️ **WARNING**: Never tune or select random seeds based on performance results.

**From Patterson et al. (2024)**: "The seed is NOT a tuneable hyperparameter"

Selecting seeds that produce favorable results invalidates your experimental results and leads to:
- Overfitting to specific random initializations
- Results that don't generalize
- Misleading conclusions about algorithm performance
- Inability for others to reproduce your results

**Good Practices**:
- ✅ Let the system generate random seeds automatically
- ✅ Run multiple seeds (30+ for strong claims) and report aggregate statistics
- ✅ Report ALL runs, including those with poor performance
- ✅ Log all seeds in experiment metadata for reproducibility

**Bad Practices**:
- ❌ Running 10 seeds and only reporting the best 5
- ❌ "Trying different seeds" until you get good results
- ❌ Choosing specific seed values because they work well
- ❌ Discarding runs as "bad seeds" without strong justification

**Reference**: Patterson et al. (2024) "Empirical Design in Reinforcement Learning"
https://arxiv.org/abs/2304.01315
```

---

### Medium Priority

#### R4: Add Individual Trajectory Logging

**Paper Justification**: "Plot single agent performance trajectories" ([Patterson et al., 2024](https://www.brandonrohrer.com/empirical_design_rl))

**Proposed**: Log individual runs to WandB with grouping for multi-seed experiments

#### R5: Add Distribution Visualization

**Paper Justification**: "Use kernel density estimation" ([Patterson et al., 2024](https://www.brandonrohrer.com/empirical_design_rl))

**Proposed**: Create KDE plots and histograms showing performance distributions

#### R6: Add Statistical Hypothesis Testing

**Paper Context**: Discusses proper hypothesis testing approaches

**Proposed**: Implement Welch's t-test, bootstrap tests, and multiple comparison corrections

---

### Low Priority

#### R7: Create Experiment Reporting Template

**Paper Justification**: "Report comprehensive implementation details" ([Patterson et al., 2024](https://arxiv.org/abs/2304.01315))

#### R8: Add Baseline Tracking System

**Paper Justification**: "Apply equivalent optimization effort to baseline algorithms" ([Patterson et al., 2024](https://www.brandonrohrer.com/empirical_design_rl))

---

## Implementation Priorities

### Phase 1: Foundation (1-2 weeks)
1. **R3: Documentation** - Add seed warning to CLAUDE.md
2. **R2: Enhanced Analysis** - Add confidence intervals and percentiles
3. **Testing** - Ensure new analysis code is well-tested

### Phase 2: Multi-Seed Support (2-3 weeks)
1. **R1: Multi-Seed Training** - Build infrastructure for N-seed experiments
2. **R4: Trajectory Logging** - Log individual runs to WandB
3. **Integration Testing** - Ensure multi-seed training works end-to-end

### Phase 3: Statistical Rigor (2-3 weeks)
1. **R5: Distribution Visualization** - Add KDE plots and distribution analysis
2. **R6: Hypothesis Testing** - Build statistical comparison tools
3. **Documentation** - Write guides on using new tools

### Phase 4: Polish (1-2 weeks)
1. **R7: Reporting Template** - Create experiment report template
2. **R8: Baseline Tracking** - Build baseline management system (optional)
3. **Final Testing** - End-to-end validation

**Total Estimated Time**: 6-10 weeks for complete implementation

---

## Quick Wins

These can be implemented immediately with minimal effort:

### QW1: Document Seed Warning (30 minutes)
Add warning to CLAUDE.md as shown in R3

### QW2: Add Median to Analysis Output (2 hours)
Modify `metta/eval/analysis.py` to include median alongside mean

### QW3: Log Seed to WandB (1 hour)
Ensure seed is logged to wandb.config for reproducibility

### QW4: Add Distribution Stats (3 hours)
Add min, max, skewness to analysis output

---

## Measuring Success

**Before** (current state):
- ❌ "Our algorithm achieves reward of 0.85"
- ❌ "Algorithm A is better than B"
- ❌ "We ran 5 seeds"

**After** (following [Patterson et al., 2024](https://arxiv.org/abs/2304.01315)):
- ✅ "Our algorithm achieves reward of 0.85 ± 0.12 (mean ± std) across 30 seeds, with 95% CI [0.81, 0.89], median 0.84, IQM 0.85"
- ✅ "Algorithm A (mean=0.85, std=0.12, n=30) significantly outperforms B (mean=0.72, std=0.15, n=30) with p=0.003, Cohen's d=0.92 (large effect)"
- ✅ "We ran 30 independent seeds, reporting all results. Performance distribution is approximately normal (skewness=0.12, kurtosis=0.08)"

---

## References

### Primary Reference

**Patterson, A., Neumann, S., White, M., & White, A. (2024).** Empirical Design in Reinforcement Learning. *Journal of Machine Learning Research, 25*(183), 1-63.

- **JMLR**: https://jmlr.org/papers/v25/23-0183.html
- **arXiv**: https://arxiv.org/abs/2304.01315
- **PDF**: https://jmlr.org/papers/volume25/23-0183/23-0183.pdf

**Summary by Brandon Rohrer**: https://www.brandonrohrer.com/empirical_design_rl

### Additional Resources

**Statistical Methods**:
- Colas, C., Sigaud, O., & Oudeyer, P. Y. (2019). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv:1806.08295. https://arxiv.org/abs/1806.08295

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. *NeurIPS 2021*. arXiv:2108.13264. https://arxiv.org/abs/2108.13264

**Best Practices**:
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning That Matters. *AAAI 2018*. arXiv:1709.06560. https://arxiv.org/abs/1709.06560

---

## Appendix: 20-Point Experimental Checklist

Based on [Patterson et al. (2024)](https://arxiv.org/abs/2304.01315), here's a checklist for running rigorous experiments:

**Sample Size & Randomness**:
- [ ] 1. Run 30+ independent seeds (not 5)
- [ ] 2. Seeds generated randomly, not manually selected
- [ ] 3. All runs reported, none discarded without justification
- [ ] 4. Different seeds for train/eval splits

**Statistical Reporting**:
- [ ] 5. Report confidence intervals or standard deviations
- [ ] 6. Report median and IQM, not just mean
- [ ] 7. Show distribution plots (KDE, violin, box)
- [ ] 8. Avoid standard errors (use confidence intervals)

**Visualization**:
- [ ] 9. Plot individual run trajectories, not just aggregates
- [ ] 10. Use interaction steps on x-axis, not episodes
- [ ] 11. Include uncertainty bands in plots
- [ ] 12. Check for outliers and investigate

**Experimental Design**:
- [ ] 13. Apply equal tuning effort to all algorithms
- [ ] 14. Document hyperparameter search process
- [ ] 15. Test on multiple environments, not cherry-picked ones
- [ ] 16. Avoid aggressive episode cutoffs

**Statistical Testing**:
- [ ] 17. Perform hypothesis tests for comparisons
- [ ] 18. Apply multiple comparison corrections
- [ ] 19. Report effect sizes (Cohen's d)
- [ ] 20. Check distribution assumptions (normality, etc.)

---

**Document History**:
- 2025-10-20: Initial version based on Patterson et al. (2024) "Empirical Design in Reinforcement Learning" (arXiv:2304.01315)

**Paper Citation**:
```bibtex
@article{patterson2024empirical,
  title={Empirical Design in Reinforcement Learning},
  author={Patterson, Andrew and Neumann, Samuel and White, Martha and White, Adam},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={183},
  pages={1--63},
  year={2024}
}
```
