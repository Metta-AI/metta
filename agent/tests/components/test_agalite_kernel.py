import torch

from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig


def test_dpfp_feature_shape():
    kernel = AGaLiTeKernelConfig(name="dpfp", nu=2)
    base = torch.randn(5, 8)
    proj = torch.randn(5, 2)

    features = kernel.feature_map(base, proj, eta=proj.shape[-1])
    assert features.shape == (5, kernel.feature_dim(base.shape[-1], proj.shape[-1]))

    gamma = torch.randn(5, 8)
    gamma_features = kernel.gamma_map(gamma, proj, eta=proj.shape[-1])
    assert gamma_features.shape == (5, kernel.feature_dim(gamma.shape[-1], proj.shape[-1]))


def test_parameterized_projection_feature_shape():
    kernel = AGaLiTeKernelConfig(name="pp_relu", nu=3)
    base = torch.randn(7, 6)
    proj = torch.randn(7, kernel.nu)

    features = kernel.feature_map(base, proj, eta=kernel.nu)
    assert features.shape == (7, kernel.feature_dim(base.shape[-1], kernel.nu))

    gamma = torch.randn(7, 6)
    gamma_features = kernel.gamma_map(gamma, proj, eta=kernel.nu)
    assert gamma_features.shape == (7, kernel.feature_dim(gamma.shape[-1], kernel.nu))


def test_gamma_map_bounded():
    kernel = AGaLiTeKernelConfig(name="eluplus1", nu=2)
    gamma = torch.randn(3, 4, 5)
    proj = torch.randn(3, 4, kernel.nu)

    gamma_features = kernel.gamma_map(gamma, proj, eta=kernel.nu)

    assert torch.all(gamma_features >= 0)
    assert torch.all(gamma_features <= 1 + 1e-6)
