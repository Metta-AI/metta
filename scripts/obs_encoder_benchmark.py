"""Benchmark different observation encoders on real MettaGrid observations.

Runs a short rollout in the arena env to capture real token observations, then
benchmarks encoder variants side-by-side:
- Perceiver (current default)
- LatentAttn (query-only cross attention)
- SelfAttn (shallow CLS self-attn)

By default runs fp16 on GPU with mask off (flash/xops path).
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch
from tensordict import TensorDict

from metta.agent.components.obs_enc import (
    ObsLatentAttn,
    ObsLatentAttnConfig,
    ObsPerceiverLatent,
    ObsPerceiverLatentConfig,
    ObsSelfAttn,
    ObsSelfAttnConfig,
)
from metta.agent.components.obs_shim import ObsShimTokens, ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrEmbedFourierConfig
from mettagrid.builder.envs import make_arena
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.simulator import Simulator


def collect_obs(sim: Simulator, steps: int) -> list[torch.Tensor]:
    """Roll out `steps` with noop actions and collect raw env_obs tokens."""
    obs_list: list[torch.Tensor] = []
    c_sim = sim._c_sim  # noqa: SLF001
    for _ in range(steps):
        obs_np = c_sim.observations().copy()  # [agents, M, 3] uint8
        obs_list.append(torch.from_numpy(obs_np))
        c_sim.actions()[:] = 0  # noop
        sim.step()
    return obs_list


def benchmark_variant(
    label: str,
    obs_sequences: list[torch.Tensor],
    shim: ObsShimTokens,
    embed: ObsAttrEmbedFourier,
    encoder: Callable[[TensorDict], TensorDict],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Benchmark a single encoder variant end-to-end (shim+embed+encoder)."""
    times: list[float] = []
    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            for obs in obs_sequences:
                td = TensorDict({"env_obs": obs.to(device)}, batch_size=[obs.shape[0]])
                td = shim(td)
                td = embed(td)
                td["obs_attr_embed"] = td["obs_attr_embed"].to(dtype)
                encoder(td)
        torch.cuda.synchronize(device=device)

        # Timed runs
        for _ in range(iters):
            for obs in obs_sequences:
                td = TensorDict({"env_obs": obs.to(device)}, batch_size=[obs.shape[0]])
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                td = shim(td)
                td = embed(td)
                td["obs_attr_embed"] = td["obs_attr_embed"].to(dtype)
                encoder(td)
                end.record()
                torch.cuda.synchronize(device=device)
                times.append(start.elapsed_time(end))

    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    p99 = statistics.quantiles(times, n=100)[98]
    print(f"{label:26s} mean={mean_ms:7.3f} ms  median={median_ms:7.3f}  p90={p90:7.3f}  p99={p99:7.3f}")
    return {"mean_ms": mean_ms, "median_ms": median_ms, "p90_ms": p90, "p99_ms": p99}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark obs encoders on real env observations.")
    parser.add_argument("--steps", type=int, default=15, help="Rollout steps to capture observations.")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations (per captured step).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (per captured step).")
    parser.add_argument("--num-agents", type=int, default=24, help="Number of agents for arena env.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--latent-tokens", type=int, default=12)
    parser.add_argument("--latent-heads", type=int, default=4)
    parser.add_argument("--latent-layers", type=int, default=2)
    parser.add_argument("--latentattn-q", type=int, default=8)
    parser.add_argument("--latentattn-layers", type=int, default=1)
    parser.add_argument("--selfattn-heads", type=int, default=3)
    parser.add_argument("--selfattn-layers", type=int, default=1)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Build env and collect observations
    cfg = make_arena(num_agents=args.num_agents)
    sim = Simulator().new_simulation(cfg, seed=0)
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

    obs_sequences = collect_obs(sim, args.steps)
    print(f"Captured {len(obs_sequences)} steps of observations; per-step shape {obs_sequences[0].shape}")

    shim = ObsShimTokens(
        policy_env_info,
        ObsShimTokensConfig(in_key="env_obs", out_key="tokens", max_tokens=args.max_tokens),
    ).to(device)

    embed = ObsAttrEmbedFourier(
        ObsAttrEmbedFourierConfig(in_key="tokens", out_key="obs_attr_embed", attr_embed_dim=8, num_freqs=3)
    ).to(device)

    feat_dim = 8 + 4 * 3 + 1  # attr_emb + fourier + value = 21

    perceiver = ObsPerceiverLatent(
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="enc",
            feat_dim=feat_dim,
            latent_dim=64,
            num_latents=args.latent_tokens,
            num_heads=args.latent_heads,
            num_layers=args.latent_layers,
            mlp_ratio=4.0,
            use_mask=False,
            pool="mean",
        )
    ).to(device=device, dtype=dtype)

    latent_attn = ObsLatentAttn(
        ObsLatentAttnConfig(
            in_key="obs_attr_embed",
            out_key="enc",
            feat_dim=feat_dim,
            out_dim=64,
            use_mask=False,
            num_query_tokens=args.latentattn_q,
            num_heads=args.latent_heads,
            num_layers=args.latentattn_layers,
            query_token_dim=64,
            qk_dim=64,
            v_dim=64,
            use_cls_token=True,
        )
    ).to(device=device, dtype=dtype)

    self_attn = ObsSelfAttn(
        ObsSelfAttnConfig(
            feat_dim=feat_dim,
            in_key="obs_attr_embed",
            out_key="enc",
            out_dim=64,
            use_mask=False,
            num_layers=args.selfattn_layers,
            num_heads=args.selfattn_heads,
            use_cls_token=True,
        )
    ).to(device=device, dtype=dtype)

    print(
        f"device={device}, dtype={dtype}, steps={args.steps}, iters={args.iters}, warmup={args.warmup}, "
        f"latents={args.latent_tokens}x{args.latent_layers}, heads={args.latent_heads}, "
        f"latentattn_q={args.latentattn_q}, latentattn_layers={args.latentattn_layers}, "
        f"selfattn_layers={args.selfattn_layers}, selfattn_heads={args.selfattn_heads}"
    )

    benchmark_variant(
        "Perceiver (default)",
        obs_sequences,
        shim,
        embed,
        perceiver,
        device,
        dtype,
        args.warmup,
        args.iters,
    )
    benchmark_variant(
        "LatentAttn (q=8,L=1)", obs_sequences, shim, embed, latent_attn, device, dtype, args.warmup, args.iters
    )
    benchmark_variant("SelfAttn (L=1)", obs_sequences, shim, embed, self_attn, device, dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
