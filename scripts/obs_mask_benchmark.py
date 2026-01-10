"""Micro-benchmark for obs encoder masking.

Compares ObsPerceiverLatent with and without an attention mask. The masked variant
passes `obs_mask` into the attention call; the unmasked variant matches the current
code path (flash/xops eligible).
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import MethodType

import torch
from einops import rearrange
from tensordict import TensorDict

from metta.agent.components.obs_enc import ObsPerceiverLatent, ObsPerceiverLatentConfig


def make_td(
    batch: int,
    tokens: int,
    feat_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorDict:
    x = torch.randn(batch, tokens, feat_dim, device=device, dtype=dtype)
    # Build a random padding mask to simulate ragged sequences.
    lengths = torch.randint(low=tokens // 3, high=tokens + 1, size=(batch,), device=device)
    positions = torch.arange(tokens, device=device).unsqueeze(0)
    mask = positions >= lengths.unsqueeze(1)
    return TensorDict({"obs_attr_embed": x, "obs_mask": mask}, batch_size=[batch])


def bind_masked_forward(model: ObsPerceiverLatent, tokens: int) -> None:
    """Wrap forward to feed obs_mask into attention when use_mask is True."""

    def forward(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        x_features = td[self.config.in_key]
        tokens_norm = self.token_norm(x_features)
        kv = self.kv_proj(tokens_norm)
        k, v = kv.split(self._latent_dim, dim=-1)
        k = rearrange(k, "b m (h d) -> b h m d", h=self._num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self._num_heads)

        latents = self.latents.expand(x_features.shape[0], -1, -1)

        attn_mask = None
        if self._use_mask and "obs_mask" in td.keys():
            attn_mask = td["obs_mask"].view(x_features.shape[0], 1, 1, tokens)

        for layer in self.layers:
            residual = latents
            q = layer["q_proj"](layer["latent_norm"](latents))
            q = rearrange(q, "b n (h d) -> b h n d", h=self._num_heads)

            attn_output = self._attention(q, k, v, attn_mask)
            attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
            latents = residual + layer["attn_out_proj"](attn_output)

            latents = latents + layer["mlp"](layer["mlp_norm"](latents))

        latents = self.final_norm(latents)

        if self._pool == "mean":
            latents = latents.mean(dim=1)
        elif self._pool == "first":
            latents = latents[:, 0]
        elif self._pool == "none":
            latents = rearrange(latents, "b n d -> b (n d)")
        else:
            raise ValueError("unsupported pool mode")

        td[self.config.out_key] = latents
        return td

    model.forward = MethodType(forward, model)


def _time_run(fn, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)  # milliseconds
    else:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0


def benchmark(model, td, label: str, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    with torch.inference_mode():
        for _ in range(warmup):
            model(td)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(iters):
            times.append(_time_run(lambda: model(td), device))
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]  # 90th percentile
    p99 = statistics.quantiles(times, n=100)[98]  # 99th percentile
    print(f"{label:32s} mean={mean_ms:7.3f} ms  median={median_ms:7.3f} ms  p90={p90:7.3f}  p99={p99:7.3f}")
    return {"mean_ms": mean_ms, "median_ms": median_ms, "p90_ms": p90, "p99_ms": p99}


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[name]


def main():
    parser = argparse.ArgumentParser(description="Benchmark ObsPerceiverLatent with/without attention mask.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--feat-dim", type=int, default=21)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-latents", type=int, default=12)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=400, help="Timing iterations (per variant).")
    parser.add_argument("--warmup", type=int, default=40, help="Warmup iterations (per variant).")
    parser.add_argument("--pool", choices=["mean", "first", "none"], default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--disable-xops", action="store_true", help="Force SDPA path by disabling xformers.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(0)

    td = make_td(args.batch_size, args.tokens, args.feat_dim, device, dtype)

    cfg = ObsPerceiverLatentConfig(
        in_key="obs_attr_embed",
        out_key="obs_latent_attn",
        feat_dim=args.feat_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_heads=args.heads,
        num_layers=args.layers,
        mlp_ratio=4.0,
        use_mask=False,
        pool=args.pool,
    )

    model_nomask = ObsPerceiverLatent(cfg).to(device=device, dtype=dtype)
    model_mask = ObsPerceiverLatent(cfg.model_copy(update={"use_mask": True})).to(device=device, dtype=dtype)
    bind_masked_forward(model_mask, tokens=args.tokens)

    if args.disable_xops:
        model_nomask._xops = None
        model_mask._xops = None

    print(
        f"device={device}, batch={args.batch_size}, tokens={args.tokens}, feat_dim={args.feat_dim}, "
        f"latent_dim={args.latent_dim}, latents={args.num_latents}, heads={args.heads}, layers={args.layers}, "
        f"iters={args.iters}, warmup={args.warmup}, disable_xops={args.disable_xops}, dtype={dtype}"
    )

    benchmark(model_nomask, td, "use_mask=False (baseline)", device, args.warmup, args.iters)
    benchmark(model_mask, td, "use_mask=True  (attn_mask)", device, args.warmup, args.iters)


if __name__ == "__main__":
    main()
