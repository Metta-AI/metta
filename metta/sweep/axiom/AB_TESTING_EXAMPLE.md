# A/B Testing with tAXIOM: A Concrete Example

## The Experimental Question

Your team has been training vision models on a custom dataset. Performance plateaued at 89% accuracy. Two researchers propose different solutions:

**Researcher A**: "The problem is overfitting. We need aggressive data augmentation."

**Researcher B**: "The problem is optimization. We need better learning rate scheduling."

Traditionally, you would test these hypotheses sequentially or create separate codebases. With tAXIOM, you can run a controlled A/B test where literally everything except the variable under test remains identical.

## Setting Up the Protocol

First, define the experimental protocol that both approaches will use:

```python
def vision_training_protocol():
    """Fixed protocol for training vision models."""
    return Pipeline()
        # Fixed data loading
        .io("load_dataset", load_standard_dataset)
        .through(checks=[dataset_size_correct(), images_normalized()])
        
        # VARIABLE: Augmentation strategy
        .require_join(
            "augmentation",
            exit_checks=[preserves_batch_size(), maintains_image_shape()]
        )
        
        # Fixed architecture
        .stage("build_model", create_resnet50)
        
        # VARIABLE: Learning rate schedule  
        .require_join(
            "lr_scheduler",
            exit_checks=[provides_learning_rate(), monotonic_decay()]
        )
        
        # Fixed training loop
        .stage("train_epoch", standard_training_loop)
        .through(checks=[loss_finite(), gradients_healthy()])
        
        # Fixed evaluation
        .io("evaluate", compute_test_metrics)
        .through(check=metrics_computed())
```

## Implementing Approach A: Aggressive Augmentation

```python
def aggressive_augmentation():
    """Hypothesis: Strong augmentation prevents overfitting."""
    return Pipeline()
        .stage("random_crop", lambda v: {
            **v, 
            "images": random_resized_crop(v["images"], scale=(0.6, 1.0))
        })
        .stage("color_jitter", lambda v: {
            **v,
            "images": color_distortion(v["images"], strength=0.5)
        })
        .stage("mixup", lambda v: {
            **v,
            "images": mixup_batch(v["images"], v["labels"], alpha=0.4)
        })
        .through(hooks=[log_augmentation_stats])

def baseline_scheduler():
    """Keep scheduler constant for this test."""
    return Pipeline()
        .stage("compute_lr", lambda v: {
            **v,
            "learning_rate": 0.001 * (0.95 ** v["epoch"])
        })
```

## Implementing Approach B: Adaptive Scheduling

```python
def baseline_augmentation():
    """Minimal augmentation - control for scheduler test."""
    return Pipeline()
        .stage("basic_flip", lambda v: {
            **v,
            "images": horizontal_flip(v["images"], p=0.5)
        })

def adaptive_scheduler():
    """Hypothesis: Adaptive LR scheduling improves convergence."""
    return Pipeline()
        .io("analyze_plateau", track_loss_history)
        .stage("compute_lr", lambda v: {
            # Reduce LR when loss plateaus
            plateau_detected = check_plateau(v["loss_history"])
            current_lr = v.get("learning_rate", 0.001)
            new_lr = current_lr * 0.5 if plateau_detected else current_lr
            return {**v, "learning_rate": new_lr}
        })
        .stage("warm_restart", lambda v: {
            # Cyclical restart every 30 epochs
            if v["epoch"] % 30 == 0:
                return {**v, "learning_rate": 0.001}
            return v
        })
```

## Running the A/B Test

```python
def run_ab_test(n_seeds=5):
    """Execute controlled A/B test with multiple seeds."""
    
    protocol = vision_training_protocol()
    
    results_a = []
    results_b = []
    
    for seed in range(n_seeds):
        # Test A: Aggressive augmentation with baseline scheduler
        pipeline_a = protocol
            .provide_join("augmentation", aggressive_augmentation())
            .provide_join("lr_scheduler", baseline_scheduler())
        
        # Test B: Baseline augmentation with adaptive scheduler
        pipeline_b = protocol
            .provide_join("augmentation", baseline_augmentation())
            .provide_join("lr_scheduler", adaptive_scheduler())
        
        # Run both with identical initialization
        set_random_seed(seed)
        result_a = pipeline_a.run()
        
        set_random_seed(seed)  # Same seed!
        result_b = pipeline_b.run()
        
        results_a.append(result_a["test_accuracy"])
        results_b.append(result_b["test_accuracy"])
    
    return {
        "augmentation_approach": {
            "mean": np.mean(results_a),
            "std": np.std(results_a),
            "values": results_a
        },
        "scheduler_approach": {
            "mean": np.mean(results_b),
            "std": np.std(results_b),
            "values": results_b
        }
    }
```

## The Critical Guarantees

What makes this a true A/B test:

1. **Identical data pipeline**: Both approaches load and preprocess data identically
2. **Identical architecture**: Same ResNet-50 initialization
3. **Identical training loop**: Same optimizer, same batch size, same number of epochs
4. **Identical evaluation**: Same test set, same metrics
5. **Identical initialization**: Same random seeds

The ONLY differences are the augmentation strategy (Test A) or the scheduler strategy (Test B).

## Analyzing Results

```python
results = run_ab_test(n_seeds=5)

# Statistical comparison
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    results["augmentation_approach"]["values"],
    results["scheduler_approach"]["values"]
)

print(f"Augmentation approach: {results['augmentation_approach']['mean']:.3f} ± "
      f"{results['augmentation_approach']['std']:.3f}")
print(f"Scheduler approach: {results['scheduler_approach']['mean']:.3f} ± "
      f"{results['scheduler_approach']['std']:.3f}")
print(f"Statistical significance: p={p_value:.4f}")

# Output:
# Augmentation approach: 0.916 ± 0.008
# Scheduler approach: 0.902 ± 0.012  
# Statistical significance: p=0.032
```

## Going Deeper: Interaction Testing

Now test if the approaches work better together:

```python
def interaction_test():
    """Test all four combinations."""
    protocol = vision_training_protocol()
    
    combinations = [
        ("baseline", "baseline", baseline_augmentation(), baseline_scheduler()),
        ("aggressive", "baseline", aggressive_augmentation(), baseline_scheduler()),
        ("baseline", "adaptive", baseline_augmentation(), adaptive_scheduler()),
        ("aggressive", "adaptive", aggressive_augmentation(), adaptive_scheduler())
    ]
    
    results = {}
    for aug_name, sched_name, aug_impl, sched_impl in combinations:
        pipeline = protocol
            .provide_join("augmentation", aug_impl)
            .provide_join("lr_scheduler", sched_impl)
        
        result = pipeline.run()
        results[(aug_name, sched_name)] = result["test_accuracy"]
    
    # Analyze interaction
    # Main effect of augmentation: 0.027 (huge)
    # Main effect of scheduler: 0.013 (moderate)
    # Interaction effect: -0.008 (slightly antagonistic)
    
    return results
```

## The Power of This Approach

Without tAXIOM, you might:
- Change both augmentation AND accidentally change the batch size
- Use different random seeds without realizing it
- Have slightly different evaluation procedures
- Not be sure if improvements came from the intended change or some other modification

With tAXIOM:
- The protocol guarantees everything except the intended variable remains constant
- Exit checks ensure the implementations are comparable
- The structure makes it obvious what's being tested
- Results are scientifically valid

This is A/B testing with actual control, not just the hope of control.