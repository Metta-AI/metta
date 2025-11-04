"""Tests for automatic alignment of vectorized environment configuration."""

from cogames.train import _resolve_vector_counts


def test_align_defaults_prefers_divisor_reduction() -> None:
    aligned_envs, aligned_workers = _resolve_vector_counts(
        256,
        10,
        envs_user_supplied=False,
        workers_user_supplied=False,
    )

    assert aligned_envs == 260
    assert aligned_workers == 10


def test_align_respects_user_envs_by_shrinking_workers() -> None:
    aligned_envs, aligned_workers = _resolve_vector_counts(
        150,
        16,
        envs_user_supplied=True,
        workers_user_supplied=False,
    )

    assert aligned_envs == 150
    assert aligned_workers == 15


def test_align_respects_user_workers_by_growing_envs() -> None:
    aligned_envs, aligned_workers = _resolve_vector_counts(
        256,
        12,
        envs_user_supplied=False,
        workers_user_supplied=True,
    )

    assert aligned_envs == 264
    assert aligned_workers == 12


def test_align_leaves_user_pair_unchanged() -> None:
    aligned_envs, aligned_workers = _resolve_vector_counts(
        50,
        12,
        envs_user_supplied=True,
        workers_user_supplied=True,
    )

    assert aligned_envs == 50
    assert aligned_workers == 12


def test_align_leaves_underfilled_worker_pair() -> None:
    aligned_envs, aligned_workers = _resolve_vector_counts(
        4,
        8,
        envs_user_supplied=True,
        workers_user_supplied=True,
    )

    assert aligned_envs == 4
    assert aligned_workers == 8
