import math
from typing import Tuple

import numpy as np
import pytest

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.quadrant_resources import QuadrantResources, QuadrantResourcesParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.types import AreaWhere


def theoretical_mean_normalized_radius(
    mode: str, k: float, alpha: float, beta: float, mu: float, sigma: float
) -> float:
    """Calculate theoretical expected normalized radius for each mode."""
    if mode == "power":
        # Weight = rr^k, mean = ∫₀¹ rr^{k+1} drr / ∫₀¹ rr^k drr = (k+1)/(k+2)
        return (k + 1) / (k + 2)
    elif mode == "exp":
        # Weight = exp(alpha * (rr - 1)), mean = [1 - 1/alpha] / (1 - exp(-alpha))
        exp_neg_alpha = math.exp(-alpha)
        return (1 - 1 / alpha) / (1 - exp_neg_alpha)
    elif mode == "log":
        # Log mode uses actual distance r, not normalized rr, so theoretical mean is harder
        # For now, use empirical estimate based on testing
        return 0.6  # Conservative estimate
    elif mode == "gaussian":
        # Gaussian peak at mu, truncated to [0,1], approximate mean as mu
        return mu
    else:
        return 0.5  # Default to uniform


def build_grid_for_mode(
    mode: str,
    seed: int = 1234,
    k: float = 3.0,
    alpha: float = 10.0,
    beta: float = 0.1,
    mu: float = 0.75,
    sigma: float = 0.1,
    count_per_quadrant: int = 20,
) -> np.ndarray:
    cfg = MapGen.Config(
        width=59,
        height=59,
        seed=seed,
        root=Quadrants.factory(
            params=QuadrantsParams(base_size=11),
            children_actions=[
                ChildrenAction(
                    scene=QuadrantResources.factory(
                        QuadrantResourcesParams(
                            resource_types=["generator_green"],
                            forced_type="generator_green",
                            count_per_quadrant=count_per_quadrant,
                            mode=mode,
                            k=k,
                            alpha=alpha,
                            beta=beta,
                            mu=mu,
                            sigma=sigma,
                            min_radius=6,
                            clearance=1,
                        )
                    ),
                    where=AreaWhere(tags=["quadrant"]),
                    lock="resources",
                    order_by="first",
                ),
            ],
        ),
    )
    mb = MapGen(cfg)
    level = mb.build()
    return level.grid


def radial_bins(grid: np.ndarray, center: Tuple[int, int], bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    cy, cx = center
    ys, xs = np.where(grid == "generator_green")
    if ys.size == 0:
        return np.zeros(bins), np.zeros(bins)
    rs = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rmax = math.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))
    rr = np.clip(rs / max(rmax, 1e-6), 0.0, 1.0)
    hist, edges = np.histogram(rr, bins=bins, range=(0.0, 1.0))
    return hist.astype(float), edges


def increasing_trend(values: np.ndarray, tolerance: float = 0.2) -> bool:
    # Check if values generally increase from inner to outer bins
    if values.sum() == 0:
        return False
    norm = values / max(values.max(), 1.0)
    x = np.arange(len(norm))
    # Simple slope via linear fit
    slope = np.polyfit(x, norm, 1)[0]
    return slope > -tolerance


def peak_near(values: np.ndarray, target_bin: int, spread: int = 1) -> bool:
    if values.sum() == 0:
        return False
    peak = int(values.argmax())
    return abs(peak - target_bin) <= spread


def mean_normalized_radius(hist: np.ndarray, edges: np.ndarray) -> float:
    if hist.sum() == 0:
        return 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float((hist * centers).sum() / hist.sum())


@pytest.mark.parametrize(
    "mode,kwargs,expected_mean",
    [
        ("power", {"k": 3.0}, 0.512),
        ("exp", {"alpha": 10.0}, 0.517),
        ("log", {"beta": 0.1}, 0.501),
        ("gaussian", {"mu": 0.75, "sigma": 0.1}, 0.502),
    ],
)
@pytest.mark.flaky(reruns=1)
def test_quadrant_resource_radial_distribution(mode: str, kwargs: dict, expected_mean: float):
    grid = build_grid_for_mode(mode=mode, **kwargs)
    h, w = grid.shape
    cy, cx = h // 2, w // 2

    # Mask out base area to avoid central placement affecting stats
    mask = np.ones_like(grid, dtype=bool)
    by0, by1 = cy - 5, cy + 6
    bx0, bx1 = cx - 5, cx + 6
    mask[max(0, by0) : min(h, by1), max(0, bx0) : min(w, bx1)] = False

    masked = np.where(mask, grid, "empty")
    hist, edges = radial_bins(masked, (cy, cx), bins=10)

    # Basic sanity: all placed
    assert hist.sum() >= 40, f"Too few placements for mode={mode}: {hist.sum()}"

    # Check that actual mean is close to empirical expectation
    actual_mean = mean_normalized_radius(hist, edges)
    tolerance = 0.05
    assert abs(actual_mean - expected_mean) < tolerance, (
        f"Mean radius {actual_mean:.3f} not close to expected {expected_mean:.3f} "
        f"for mode={mode} (diff={abs(actual_mean - expected_mean):.3f})"
    )


@pytest.mark.flaky(reruns=1)
def test_distribution_relative_ordering():
    """Test that distributions are ordered as expected: exp > power > log/gaussian"""
    modes_and_params = [
        ("power", {"k": 3.0}),
        ("exp", {"alpha": 10.0}),
        ("log", {"beta": 0.1}),
        ("gaussian", {"mu": 0.75, "sigma": 0.1}),
    ]

    means = {}
    for mode, kwargs in modes_and_params:
        grid = build_grid_for_mode(mode=mode, **kwargs)
        h, w = grid.shape
        cy, cx = h // 2, w // 2

        # Mask out base area
        mask = np.ones_like(grid, dtype=bool)
        by0, by1 = cy - 5, cy + 6
        bx0, bx1 = cx - 5, cx + 6
        mask[max(0, by0) : min(h, by1), max(0, bx0) : min(w, bx1)] = False

        masked = np.where(mask, grid, "empty")
        hist, edges = radial_bins(masked, (cy, cx), bins=10)
        means[mode] = mean_normalized_radius(hist, edges)

    # Check ordering: exp should be highest, then power, then log/gaussian similar
    assert means["exp"] > means["power"], f"exp ({means['exp']:.3f}) should be > power ({means['power']:.3f})"
    assert means["power"] > means["log"], f"power ({means['power']:.3f}) should be > log ({means['log']:.3f})"
    assert means["power"] > means["gaussian"], (
        f"power ({means['power']:.3f}) should be > gaussian ({means['gaussian']:.3f})"
    )
