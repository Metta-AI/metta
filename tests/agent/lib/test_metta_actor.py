from time import perf_counter

import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.actor import MettaActorBigModular
from metta.agent.lib.metta_module import MettaDict, MettaModule
from metta.agent.lib.modular_network import ModularNetwork


# ---- Unit Tests ----
def test_actor_big_modular_init_and_shape_inference():
    hidden_dim = 8
    num_actions = 5
    embed_dim = 4
    actor = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
    )
    assert actor.in_keys == ["hidden", "action_embeds"]
    assert actor.out_keys == ["logits"]
    assert actor.input_features_shape == [[hidden_dim], [num_actions, embed_dim]]
    assert actor.infer_output_shape([[hidden_dim], [num_actions, embed_dim]]) == [num_actions]
    assert actor.hidden_dim == hidden_dim
    assert actor.embed_dim == embed_dim


def test_actor_big_modular_forward():
    batch = 3
    hidden_dim = 8
    num_actions = 5
    embed_dim = 4
    hidden = torch.randn(batch, hidden_dim)
    action_embeds = torch.randn(batch, num_actions, embed_dim)
    actor = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
    )
    md = MettaDict(TensorDict({"hidden": hidden, "action_embeds": action_embeds}, batch_size=[batch]), {})
    out = actor(md)
    assert "logits" in out.td
    assert out.td["logits"].shape == (batch, num_actions)


def test_actor_big_modular_shape_error():
    # Wrong input shape should raise
    hidden_dim = 8
    num_actions = 5
    embed_dim = 4
    actor = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
    )
    # Wrong shape for action_embeds
    hidden = torch.randn(2, hidden_dim)
    action_embeds = torch.randn(2, num_actions, embed_dim + 1)
    md = MettaDict(TensorDict({"hidden": hidden, "action_embeds": action_embeds}, batch_size=[2]), {})
    with pytest.raises(Exception):
        actor(md)


# ---- Integration Test ----
class DummyInputModule(MettaModule):
    def __init__(self, out_key, out_shape):
        super().__init__(in_keys=[], out_keys=[out_key], input_features_shape=None, output_features_shape=out_shape)
        self.out_key = out_key
        self.out_shape = out_shape

    def _compute(self, md):
        batch = md.data.get("batch", 2)
        return {self.out_key: torch.ones((batch, *self.out_shape))}


def test_actor_big_modular_in_modular_network():
    batch = 2
    hidden_dim = 8
    num_actions = 5
    embed_dim = 4
    # Build network
    net = ModularNetwork()
    net.add_component("hidden", DummyInputModule("hidden", [hidden_dim]))
    net.add_component("action_embeds", DummyInputModule("action_embeds", [num_actions, embed_dim]))
    actor = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
    )
    net.add_component("actor", actor)
    # Prepare input
    md = MettaDict(TensorDict({}, batch_size=[batch]), {"batch": batch})
    out = net(md)
    assert "logits" in out.td
    assert out.td["logits"].shape == (batch, num_actions)


# ---- Performance Test ----
def test_actor_big_modular_performance():
    batch = 128
    hidden_dim = 64
    num_actions = 32
    embed_dim = 16
    hidden = torch.randn(batch, hidden_dim)
    action_embeds = torch.randn(batch, num_actions, embed_dim)
    actor = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
    )
    md = MettaDict(TensorDict({"hidden": hidden, "action_embeds": action_embeds}, batch_size=[batch]), {})
    # Warmup
    for _ in range(5):
        _ = actor(md)
    # Timing
    n = 100
    start = perf_counter()
    for _ in range(n):
        _ = actor(md)
    elapsed = perf_counter() - start
    avg_ms = (elapsed / n) * 1000
    print(f"MettaActorBigModular avg forward pass: {avg_ms:.3f} ms (batch={batch}, num_actions={num_actions})")
    # Just check it's not absurdly slow
    assert avg_ms < 50  # Should be fast on modern hardware


def test_actor_big_modular_vs_legacy_performance():
    from metta.agent.lib.actor import MettaActorBig

    batch = 128
    hidden_dim = 64
    num_actions = 32
    embed_dim = 16
    bilinear_output_dim = 32
    mlp_hidden_dim = 64
    hidden = torch.randn(batch, hidden_dim)
    action_embeds = torch.randn(batch, num_actions, embed_dim)
    # Legacy
    legacy = MettaActorBig(bilinear_output_dim=bilinear_output_dim, mlp_hidden_dim=mlp_hidden_dim)
    legacy._in_tensor_shapes = [[hidden_dim], [num_actions, embed_dim]]
    legacy._name = "logits"
    legacy._sources = [{"name": "hidden"}, {"name": "action_embeds"}]
    legacy._make_net()
    td_legacy = TensorDict({"hidden": hidden, "action_embeds": action_embeds}, batch_size=[batch])
    # Modular
    modular = MettaActorBigModular(
        in_keys=["hidden", "action_embeds"],
        out_keys=["logits"],
        input_features_shape=[[hidden_dim], [num_actions, embed_dim]],
        bilinear_output_dim=bilinear_output_dim,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    # Copy weights for fair comparison
    modular.W.data.copy_(legacy.W.data)
    modular.bias.data.copy_(legacy.bias.data)
    for idx in [0, 2]:
        m_mod = modular._MLP[idx]
        m_leg = legacy._MLP[idx]
        if (
            hasattr(m_mod, "weight")
            and isinstance(m_mod.weight, torch.Tensor)
            and hasattr(m_leg, "weight")
            and isinstance(m_leg.weight, torch.Tensor)
        ):
            m_mod.weight.data.copy_(m_leg.weight.data)
        if (
            hasattr(m_mod, "bias")
            and isinstance(m_mod.bias, torch.Tensor)
            and hasattr(m_leg, "bias")
            and isinstance(m_leg.bias, torch.Tensor)
        ):
            m_mod.bias.data.copy_(m_leg.bias.data)
    md_modular = MettaDict(TensorDict({"hidden": hidden, "action_embeds": action_embeds}, batch_size=[batch]), {})
    # Warmup
    for _ in range(5):
        _ = legacy._forward(td_legacy)
        _ = modular(md_modular)
    # Timing
    n = 100
    start_legacy = perf_counter()
    for _ in range(n):
        _ = legacy._forward(td_legacy)
    elapsed_legacy = perf_counter() - start_legacy
    avg_ms_legacy = (elapsed_legacy / n) * 1000
    start_modular = perf_counter()
    for _ in range(n):
        _ = modular(md_modular)
    elapsed_modular = perf_counter() - start_modular
    avg_ms_modular = (elapsed_modular / n) * 1000
    timing_msg = (
        f"\nLegacy MettaActorBig avg forward pass: {avg_ms_legacy:.3f} ms\n"
        f"Modular MettaActorBigModular avg forward pass: {avg_ms_modular:.3f} ms\n"
    )
    print(timing_msg)
    if avg_ms_modular >= 2.0 * avg_ms_legacy:
        raise AssertionError(timing_msg + "Modular actor is more than 2x slower than legacy!")
