# Goal

Determine whether the `cvc_random_maps` and `assembly_lines` environments **require memory** for agents to learn effectively and whether we can **show learning** and **show convergence** and estimate how long it takes to train roughly in each environment and determine if we need to adjust any environment knobs.

Architectures to test include

1. **Memory-free baseline** (XL attention with `mem_len=0`) - no state across timesteps
2. **Memory-enabled architectures** (XL with `mem_len=128`, sLSTM, AgSA) - can remember across timesteps


# Results

XL with no memory:

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.cvc_random_maps_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-12-31.memory-needed.cvc_random_maps_xl_memlen0_2b \
    trainer.total_timesteps=2000000000 \
    pattern=X mem_len=0
```

- Finished 2b steps in 1.43 hours (390k sps)
- Zero reward and hearts created, entropy stayed random (3.04 -> 3.04)


# Next up

### 🚀 To Run: XL with Memory (mem_len=128)

Direct comparison - same architecture but with cross-step memory.

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.cvc_random_maps_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-12-31.memory-needed.cvc_random_maps_xl_memlen128_2b \
    trainer.total_timesteps=2000000000 \
    pattern=X mem_len=128
```

---

### 🚀 To Run: sLSTM (Recurrent Memory)

Pure recurrent architecture with structured LSTM.

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.cvc_random_maps_train.train \
    --gpus 8 --skip-git-check \
    --run=yatharth.2025-12-31.memory-needed.cvc_random_maps_sLSTM_2b \
    trainer.total_timesteps=2000000000 \
    pattern=S
```

---

### 🚀 To Run: AgSA (Proven Combo)

AGaLiTe + sLSTM + Axon - known to work well on CVC.

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.cvc_random_maps_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-12-31.memory-needed.cvc_random_maps_AgSA_2b \
    trainer.total_timesteps=2000000000 \
    pattern=AgSA
```

---

# Assembly Lines

Same memory architecture comparison on the assembly_lines environment.

### 🚀 To Run: XL with No Memory (Baseline)

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-01-01.memory-needed.assembly_lines_xl_memlen0_2b \
    trainer.total_timesteps=2000000000 \
    pattern=X mem_len=0
```

### 🚀 To Run: XL with Memory (mem_len=128)

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-01-01.memory-needed.assembly_lines_xl_memlen128_2b \
    trainer.total_timesteps=2000000000 \
    pattern=X mem_len=128
```

### 🚀 To Run: sLSTM (Recurrent Memory)

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-01-01.memory-needed.assembly_lines_sLSTM_2b \
    trainer.total_timesteps=2000000000 \
    pattern=S
```

### 🚀 To Run: AgSA (Proven Combo)

```bash
./devops/skypilot/launch.py \
    recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \
    --gpus=8 --skip-git-check \
    --run=yatharth.2025-01-01.memory-needed.assembly_lines_AgSA_2b \
    trainer.total_timesteps=2000000000 \
    pattern=AgSA
```
