# Goal

I want to kick off same SkyPilot job on a 1x H100, 2x H100, 1x L4, 4x L4, 8x L4 and see how long they respectively take, I guess we don't have to wait for it to finish, we can check SPS (samples per second) and that is an exact proxy for when training would end? Since they are in theory doing exact same computations. Caveats: Different setups might be using different batch sizes, which could lead to different convergence. Also, there might be startup time. 20 million timesteps should be enough to let everything get warmed up over 10–15 minutes and then check SPS in WandB. Also turns out we can only kick off SkyPilot jobs on L4 GPUs, not H100.

My colleague said we have to ake pre-reservations for the H100 machines and someone someone added support for reservations to the sandbox launching script, but managed jobs can't make use of reservations. So I am running everthng on clusters instead of jobs.

# Methods

You might have to adjust this below as I renamed & moved the script.

```
./devops/run.sh
    recipes.experiment.cvc.cvc_random_maps.train
    run=yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xH100
    trainer.total_timesteps=20000000
    stack_builder=cortex.stacks.build_cortex_auto_config
    'stack_cfg={"pattern": "AgSA", "d_hidden": 128, "num_layers": 3}'
```

First I tried with 20 million steps, though it ended too quickly. But run names:

```
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xL4
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.4xL4
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.8xL4
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xH100
```

Then I tried with 2 billion steps.

```
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xL4.2b
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.4xL4.2b
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.8xL4.2b
yatharth.2025-12-30.gpu-timing.cvc_random_maps.AgSA.1xH100.2b
```

# Results

I have the following sandboxes running:

| Sandbox            | GPU    | SPS  | Speedup  | On-Demand            | Spot    |
| ------------------ | ------ | ---- | -------- | -------------------- | ------- |
| yatharth-sandbox-1 | L4:1   | 32k  | baseline | ~$1/hr               |         |
| yatharth-sandbox-2 | L4:4   | 131k | 4x       | ~$5/hr               |         |
| yatharth-sandbox-3 | L4:8   | 258k | 8x       | ~\$10/h              | ~$3/hr  |
| h100-h1            | H100:1 | 59k  | 1.8x     | ~$5/hr               |         |

Our SkyPilot jobs only support the L4 GPUs, not H100. But that's okay. 8x L4 is faster than a single H100, so that's generally the GPU configuration we'll want to use.
