from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import scipy.stats as st  # type: ignore
import wandb
from metta.common.tool import Tool
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from pydantic import BaseModel, Field, model_validator
from wandb.apis.public.runs import Run


class RunPair(BaseModel):
    control: str
    candidate: str
    seed: int | None = None


class SummarySpec(BaseModel):
    type: Literal["auc", "eval_last_n"] = "auc"

    # auc
    step_min: int | None = None
    step_max: int | None = None
    normalize_steps: bool = True
    percent: float | None = 0.25  # default to last 25% window
    percent_window: Literal["first", "last"] = "last"

    # eval_last_n
    n: int = 5

    @model_validator(mode="after")
    def _validate(self) -> "SummarySpec":
        if self.type == "auc":
            if self.percent is not None and not (0 < float(self.percent) <= 1):
                raise ValueError("summary.percent must be in (0,1] when used for AUC")
        if self.type == "eval_last_n":
            if self.n <= 0:
                raise ValueError("summary.n must be > 0 for eval_last_n")
        return self


class FetchSpec(BaseModel):
    samples: int | None = 2000
    min_step: int | None = None
    max_step: int | None = None
    keys: list[str] | None = None  # will default to [metric_key]


class BootstrapSpec(BaseModel):
    n_resamples: int = 10000
    alpha: float = 0.05
    side: Literal["two-sided", "greater", "less"] = "two-sided"
    method: Literal["bca", "percentile"] = "bca"


class TTestSpec(BaseModel):
    enabled: bool = False
    assumption_alpha: float = (
        0.05  # this is for normality testing, not for the t-test itself
    )


class PowerSpec(BaseModel):
    enabled: bool = False
    target_effect_size: float = 0.2
    beta: float = 0.8


class OutputSpec(BaseModel):
    print_table: bool = True
    csv_path: str | None = None


@dataclass
class _RunSeries:
    run_id: str
    steps: np.ndarray
    values: np.ndarray
    has_real_step: bool


def get_run(
    run_id: str,
    entity: str = METTA_WANDB_ENTITY,
    project: str = METTA_WANDB_PROJECT,
) -> Run | None:
    try:
        api = wandb.Api(timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"Error connecting to W&B: {str(e)}")
        print("Make sure you are connected to W&B: `metta status`")
        return None

    try:
        return api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:  # noqa: BLE001
        print(f"Error getting run {run_id}: {str(e)}")
        return None


def _resolve_step_column(df: pd.DataFrame) -> np.ndarray:
    if "_step" in df.columns:
        return df["_step"].to_numpy(dtype=float)
    if "step" in df.columns:
        return df["step"].to_numpy(dtype=float)
    # Fallback to index
    return np.arange(len(df), dtype=float)


def _fetch_series(run_id: str, metric_key: str, fetch: FetchSpec) -> _RunSeries:
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    keys = fetch.keys or [metric_key]

    if fetch.samples is None:
        # Full scan path with step filtering
        try:
            records = list(
                run.scan_history(
                    keys=keys, min_step=fetch.min_step, max_step=fetch.max_step
                )  # type: ignore[attr-defined]
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to scan history for run {run_id}: {e}") from e
        df = pd.DataFrame(records)
    else:
        try:
            # If a sampling budget is provided together with a step window, adaptively
            # request enough global samples to obtain approximately `samples` points
            # inside the window, then uniformly downsample the window to the target.
            if (
                fetch.min_step is not None
                and fetch.max_step is not None
                and fetch.samples is not None
            ):
                desired = max(1, int(fetch.samples))
                samples_n = max(desired, 2000)
                max_samples_cap = 10_000
                max_attempts = 3

                df_window: pd.DataFrame | None = None
                last_window: pd.DataFrame | None = None

                for _ in range(max_attempts):
                    df_all = run.history(samples=samples_n, keys=keys, pandas=True)  # type: ignore[assignment]
                    if ("_step" not in df_all.columns) and (
                        "step" not in df_all.columns
                    ):
                        raise ValueError(
                            "Requested step window but no step column ('_step' or 'step') is present in fetched history."
                        )
                    steps_all = _resolve_step_column(df_all)
                    lo = float(fetch.min_step)
                    hi = float(fetch.max_step)
                    mask = (steps_all >= lo) & (steps_all <= hi)
                    df_w = df_all.loc[mask]
                    last_window = df_w
                    if len(df_w) >= desired or samples_n >= max_samples_cap:
                        df_window = df_w
                        break
                    # Increase request size based on observed fraction in window
                    window_count = max(1, int(len(df_w)))
                    samples_n = int(
                        min(
                            max_samples_cap,
                            math.ceil(desired * samples_n / window_count * 1.25),
                        )
                    )

                if df_window is None:
                    df_window = last_window if last_window is not None else df_all

                # Fail fast if the requested window does not meet the desired budget
                if len(df_window) < desired:
                    raise ValueError(
                        f"Insufficient points in requested step window (need >= {desired}) in run {run_id}"
                    )

                # Uniformly downsample to the desired budget inside the window
                if len(df_window) > desired:
                    idx = np.linspace(0, len(df_window) - 1, num=desired, dtype=int)
                    df_window = df_window.iloc[idx]

                df = df_window
            else:
                # No window specified; regular global sampling is fine
                df = run.history(samples=fetch.samples, keys=keys, pandas=True)  # type: ignore[assignment]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to fetch sampled history for run {run_id}: {e}"
            ) from e

    if metric_key not in df.columns:
        raise ValueError(f"Metric '{metric_key}' not found in run {run_id}")

    has_real_step = ("_step" in df.columns) or ("step" in df.columns)
    steps = _resolve_step_column(df)
    values = df[metric_key].to_numpy(dtype=float)

    # raise value error if any values are NaN
    if np.isnan(values).any():
        raise ValueError(f"NaN values for metric '{metric_key}' in run {run_id}")
    return _RunSeries(
        run_id=run_id, steps=steps, values=values, has_real_step=has_real_step
    )


def _reduce_summary(series: _RunSeries, summary: SummarySpec) -> float:
    s = summary
    steps = series.steps
    vals = series.values

    if s.type == "eval_last_n":
        k = max(1, int(s.n))
        return float(np.mean(vals[-k:]))

    # AUC
    if s.percent is not None:
        k = max(1, int(math.ceil(float(s.percent) * len(vals))))
        if s.percent_window == "first":
            steps_w = steps[:k]
            vals_w = vals[:k]
        else:
            steps_w = steps[-k:]
            vals_w = vals[-k:]
    else:
        # Apply step bounds if provided; require both bounds if any is set
        if not series.has_real_step:
            raise ValueError(
                "summary.step_min/step_max requested but no step column ('_step' or 'step') is present"
            )
        if (s.step_min is None) and (s.step_max is None):
            steps_w = steps
            vals_w = vals
        elif (s.step_min is None) != (s.step_max is None):
            raise ValueError(
                "Both summary.step_min and summary.step_max must be provided together"
            )
        else:
            lo = float(s.step_min)
            hi = float(s.step_max)
            mask = (steps >= lo) & (steps <= hi)
            steps_w = steps[mask]
            vals_w = vals[mask]

    if len(steps_w) < 15:  # this is kind of arbitrary
        raise ValueError(
            f"Insufficient points in window for AUC (need >= 15) in run {series.run_id}"
        )

    auc = float(np.trapz(vals_w, steps_w))
    # below, we normalize by the duration of steps: this makes sense if the denominator is the same for all.
    if s.normalize_steps:
        if not series.has_real_step:
            raise ValueError(
                "summary.normalize_steps=True requires a real step column ('_step' or 'step')"
            )
        duration = float(steps_w[-1] - steps_w[0])
        if duration <= 0:
            raise ValueError(f"Non-positive duration for window in run {series.run_id}")
        auc /= duration
    return auc


def _mean_confidence_interval_bca(
    samples: np.ndarray,  # bootstrap statistics
    t_hat: float,
    alpha: float,
    jackknife_stats: Optional[np.ndarray] = None,
    side: Literal["two-sided", "greater", "less"] = "two-sided",
) -> tuple[float, float]:
    # Bias-correction
    eps = 1e-12
    frac = float(np.mean(samples < t_hat))
    frac = min(max(frac, eps), 1 - eps)
    z0 = NormalDist().inv_cdf(frac)

    # Acceleration via jackknife if provided, else 0
    a = 0.0
    if jackknife_stats is not None and len(jackknife_stats) > 1:
        t_dot = float(np.mean(jackknife_stats))
        diffs = t_dot - jackknife_stats
        num = float(np.sum(diffs**3))
        den = float(6.0 * (np.sum(diffs**2) ** 1.5) + eps)
        if den != 0.0:
            a = num / den

    def pct(point: float) -> float:
        dist = NormalDist()
        zalpha = dist.inv_cdf(point)
        adj = z0 + (zalpha / (1 - a * (zalpha - z0) + eps))
        return float(dist.cdf(adj))

    if side == "two-sided":
        lower_pct = pct(alpha / 2)
        upper_pct = pct(1 - alpha / 2)
        return (
            float(np.quantile(samples, lower_pct)),
            float(np.quantile(samples, upper_pct)),
        )
    if side == "greater":
        lower_pct = pct(alpha)
        return (float(np.quantile(samples, lower_pct)), float("inf"))
    # side == "less"
    upper_pct = pct(1 - alpha)
    return (float("-inf"), float(np.quantile(samples, upper_pct)))


def _percentile_ci(
    samples: np.ndarray,
    alpha: float,
    side: Literal["two-sided", "greater", "less"] = "two-sided",
) -> tuple[float, float]:
    if side == "two-sided":
        lo = float(np.quantile(samples, alpha / 2))
        hi = float(np.quantile(samples, 1 - alpha / 2))
        return lo, hi
    if side == "greater":
        lo = float(np.quantile(samples, alpha))
        return lo, float("inf")
    # side == "less"
    hi = float(np.quantile(samples, 1 - alpha))
    return float("-inf"), hi


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int,
    alpha: float,
    method: Literal["bca", "percentile"],
    side: Literal["two-sided", "greater", "less"] = "two-sided",
) -> tuple[float, float, np.ndarray]:
    n = len(diffs)
    if n == 0:
        raise ValueError("No paired differences to analyze")
    t_hat = float(np.mean(diffs))
    idx = np.random.randint(0, n, size=(n_resamples, n))
    boot = np.mean(diffs[idx], axis=1)

    if method == "bca":
        # Jackknife leave-one-out means
        jack = np.array([float(np.mean(np.delete(diffs, i))) for i in range(n)])
        # CI via BCa
        lo, hi = _mean_confidence_interval_bca(
            boot, t_hat, alpha, jackknife_stats=jack, side=side
        )
        # Bias-corrected point estimate via BCa median
        eps = 1e-12
        frac = float(np.mean(boot < t_hat))
        frac = min(max(frac, eps), 1 - eps)
        z0 = NormalDist().inv_cdf(frac)
        a = 0.0
        if len(jack) > 1:
            t_dot = float(np.mean(jack))
            diffs_j = t_dot - jack
            num = float(np.sum(diffs_j**3))
            den = float(6.0 * (np.sum(diffs_j**2) ** 1.5) + eps)
            if den != 0.0:
                a = num / den
        dist = NormalDist()
        zalpha = dist.inv_cdf(0.5)
        adj = z0 + (zalpha / (1 - a * (zalpha - z0) + eps))
        median_pct = float(dist.cdf(adj))
        point = float(np.quantile(boot, median_pct))
    else:
        lo, hi = _percentile_ci(boot, alpha, side=side)
        point = t_hat
    return point, (lo, hi), boot


def _unpaired_bootstrap_ci(
    control: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int,
    alpha: float,
    method: Literal["bca", "percentile"],
    side: Literal["two-sided", "greater", "less"] = "two-sided",
) -> tuple[float, tuple[float, float], np.ndarray]:
    if len(control) == 0 or len(candidate) == 0:
        raise ValueError("Control and candidate must have at least one sample each")
    t_hat = float(np.mean(candidate) - np.mean(control))
    idx_c = np.random.randint(0, len(control), size=(n_resamples, len(control)))
    idx_t = np.random.randint(0, len(candidate), size=(n_resamples, len(candidate)))
    boot = np.mean(candidate[idx_t], axis=1) - np.mean(control[idx_c], axis=1)

    if method == "bca":
        # Jackknife: leave-one-out across both groups
        jack_vals: list[float] = []
        for i in range(len(control)):
            jack_vals.append(float(np.mean(candidate) - np.mean(np.delete(control, i))))
        for j in range(len(candidate)):
            jack_vals.append(float(np.mean(np.delete(candidate, j)) - np.mean(control)))
        jack = np.array(jack_vals)
        # CI via BCa
        lo, hi = _mean_confidence_interval_bca(
            boot, t_hat, alpha, jackknife_stats=jack, side=side
        )
        # Bias-corrected point estimate via BCa median
        eps = 1e-12
        frac = float(np.mean(boot < t_hat))
        frac = min(max(frac, eps), 1 - eps)
        z0 = NormalDist().inv_cdf(frac)
        a = 0.0
        if len(jack) > 1:
            t_dot = float(np.mean(jack))
            diffs_j = t_dot - jack
            num = float(np.sum(diffs_j**3))
            den = float(6.0 * (np.sum(diffs_j**2) ** 1.5) + eps)
            if den != 0.0:
                a = num / den
        dist = NormalDist()
        zalpha = dist.inv_cdf(0.5)
        adj = z0 + (zalpha / (1 - a * (zalpha - z0) + eps))
        median_pct = float(dist.cdf(adj))
        point = float(np.quantile(boot, median_pct))
    else:
        lo, hi = _percentile_ci(boot, alpha, side=side)
        point = t_hat
    return point, (lo, hi), boot


def _ttest_optional(
    paired: bool, control: np.ndarray, candidate: np.ndarray, assumption_alpha: float
) -> dict[str, Any]:
    result: dict[str, Any] = {"warnings": [], "assumptions": {}}
    try:
        if paired:
            if len(control) != len(candidate):
                raise ValueError(
                    "paired t-test requires equal-length samples (control vs candidate)"
                )
            diffs = candidate - control
            # Assumptions: normality on differences
            if len(diffs) < 2:
                result["warnings"].append(
                    "Insufficient samples for Shapiro-Wilk (n<2); skipping t-test"
                )
                return result
            w_stat, w_p = st.shapiro(diffs)
            # Compute test statistics but only report if assumptions pass
            t_stat, p_val = st.ttest_rel(candidate, control, nan_policy="raise")  # type: ignore
            if float(w_p) < assumption_alpha:
                result["warnings"].append(
                    f"Shapiro-Wilk normality on diffs failed (p={float(w_p):.6g})"
                )
            else:
                result["ttest"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
            result["assumptions"].update(
                {
                    "normality_shapiro_W": float(w_stat),
                    "normality_p": float(w_p),
                }
            )
        else:
            # Assumptions: approximate normality of each group; report Levene as well
            if len(control) < 2 or len(candidate) < 2:
                result["warnings"].append(
                    "Insufficient samples for Shapiro-Wilk (n<2) in one or both groups; skipping t-test"
                )
                return result
            w_stat_c, w_p_c = st.shapiro(control)
            w_stat_t, w_p_t = st.shapiro(candidate)
            lev_stat, lev_p = st.levene(control, candidate, center="mean")
            # Compute Welch's t-test but only report if normality checks pass
            t_stat, p_val = st.ttest_ind(
                candidate, control, equal_var=False, nan_policy="raise"
            )
            violations: list[str] = []
            if float(w_p_c) < assumption_alpha:
                violations.append(f"control normality (p={float(w_p_c):.6g})")
            if float(w_p_t) < assumption_alpha:
                violations.append(f"candidate normality (p={float(w_p_t):.6g})")
            if not violations:
                result["ttest"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
            else:
                result["warnings"].append(
                    "Assumption violations: " + "; ".join(violations)
                )
            result["assumptions"].update(
                {
                    "normality_control_W": float(w_stat_c),
                    "normality_control_p": float(w_p_c),
                    "normality_candidate_W": float(w_stat_t),
                    "normality_candidate_p": float(w_p_t),
                    "levene_stat": float(lev_stat),
                    "levene_p": float(lev_p),
                }
            )
        return result
    except Exception as e:  # noqa: BLE001
        # Do not fail the overall analysis; record the failure and continue
        result["warnings"].append(f"t-test computation failed: {e}")
        return result


class CompareTool(Tool):
    # Inputs
    control_run_ids: list[str] | None = Field(default=None)
    candidate_run_ids: list[str] | None = Field(default=None)
    pairs: list[RunPair] | None = Field(default=None)

    # Analysis config
    metric_key: str = Field(default="overview/reward")
    summary: SummarySpec = Field(default_factory=SummarySpec)
    fetch: FetchSpec = Field(default_factory=FetchSpec)
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)
    ttest: TTestSpec = Field(default_factory=TTestSpec)
    power: PowerSpec = Field(default_factory=PowerSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="after")
    def _check_inputs(self) -> "CompareTool":
        if self.pairs is not None:
            if self.control_run_ids is not None or self.candidate_run_ids is not None:
                raise ValueError(
                    "Provide either pairs or control/candidate run lists, not both"
                )
            if len(self.pairs) == 0:
                raise ValueError("pairs cannot be empty")
        else:
            if not self.control_run_ids or not self.candidate_run_ids:
                raise ValueError(
                    "control_run_ids and candidate_run_ids are required for unpaired analysis"
                )
        return self

    # ----------------------------------------------------------------------------------
    # Invoke
    # ----------------------------------------------------------------------------------
    def invoke(self, args: dict[str, str]) -> int | None:
        # Determine which runs to fetch and how to summarize them
        fetch_spec = self.fetch.model_copy()
        # Default keys to fetch only what's required
        if not fetch_spec.keys:
            # Request step columns explicitly along with the metric
            fetch_spec.keys = ["_step", "step", self.metric_key]

        if self.pairs is not None:
            # Paired analysis
            control_vals: list[float] = []
            candidate_vals: list[float] = []
            pair_labels: list[str] = []

            for p in self.pairs:
                s_control = _fetch_series(p.control, self.metric_key, fetch_spec)
                s_candidate = _fetch_series(p.candidate, self.metric_key, fetch_spec)
                v_control = _reduce_summary(s_control, self.summary)
                v_candidate = _reduce_summary(s_candidate, self.summary)
                control_vals.append(v_control)
                candidate_vals.append(v_candidate)
                pair_labels.append(f"{p.control} vs {p.candidate}")

            control_arr = np.asarray(control_vals, dtype=float)
            candidate_arr = np.asarray(candidate_vals, dtype=float)
            diffs = candidate_arr - control_arr

            point, (ci_lo, ci_hi), boot = _paired_bootstrap_ci(
                diffs=diffs,
                n_resamples=self.bootstrap.n_resamples,
                alpha=self.bootstrap.alpha,
                method=(
                    "percentile" if self.bootstrap.method == "percentile" else "bca"
                ),
                side=self.bootstrap.side,
            )

            ttest_result = (
                _ttest_optional(
                    True, control_arr, candidate_arr, self.ttest.assumption_alpha
                )
                if self.ttest.enabled
                else None
            )
            power_info = (
                self._power_info(len(diffs), np.std(diffs, ddof=1))
                if self.power.enabled
                else None
            )

            self._print_results(
                paired=True,
                n_control=len(control_arr),
                n_candidate=len(candidate_arr),
                point=point,
                ci=(ci_lo, ci_hi),
                boot=boot,
                ttest=ttest_result,
                power=power_info,
            )

            if self.output.csv_path:
                self._write_csv(
                    path=self.output.csv_path,
                    summaries=[
                        ("paired", lbl, float(c), float(t))
                        for lbl, c, t in zip(
                            pair_labels, control_arr, candidate_arr, strict=False
                        )
                    ],
                    point=point,
                    ci=(ci_lo, ci_hi),
                )

        else:
            # Unpaired analysis
            control_vals = [
                _reduce_summary(
                    _fetch_series(rid, self.metric_key, fetch_spec), self.summary
                )
                for rid in self.control_run_ids or []
            ]
            candidate_vals = [
                _reduce_summary(
                    _fetch_series(rid, self.metric_key, fetch_spec), self.summary
                )
                for rid in self.candidate_run_ids or []
            ]

            control_arr = np.asarray(control_vals, dtype=float)
            candidate_arr = np.asarray(candidate_vals, dtype=float)

            point, (ci_lo, ci_hi), boot = _unpaired_bootstrap_ci(
                control=control_arr,
                candidate=candidate_arr,
                n_resamples=self.bootstrap.n_resamples,
                alpha=self.bootstrap.alpha,
                method=(
                    "percentile" if self.bootstrap.method == "percentile" else "bca"
                ),
                side=self.bootstrap.side,
            )

            ttest_result = (
                _ttest_optional(
                    False, control_arr, candidate_arr, self.ttest.assumption_alpha
                )
                if self.ttest.enabled
                else None
            )
            pooled_std = float(
                np.sqrt(
                    np.var(control_arr, ddof=1) / len(control_arr)
                    + np.var(candidate_arr, ddof=1) / len(candidate_arr)
                )
            )
            power_info = (
                self._power_info(len(control_arr) + len(candidate_arr), pooled_std)
                if self.power.enabled
                else None
            )

            self._print_results(
                paired=False,
                n_control=len(control_arr),
                n_candidate=len(candidate_arr),
                point=point,
                ci=(ci_lo, ci_hi),
                boot=boot,
                ttest=ttest_result,
                power=power_info,
            )

            if self.output.csv_path:
                self._write_csv(
                    path=self.output.csv_path,
                    summaries=[
                        ("control", rid, float(val))
                        for rid, val in zip(
                            self.control_run_ids or [], control_arr, strict=False
                        )
                    ]
                    + [
                        ("candidate", rid, float(val))
                        for rid, val in zip(
                            self.candidate_run_ids or [], candidate_arr, strict=False
                        )
                    ],
                    point=point,
                    ci=(ci_lo, ci_hi),
                )

        return 0

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _alpha_for_side(self) -> float:
        # Always return the user-specified alpha; one-sided handling is done in CI routines
        return self.bootstrap.alpha

    def _print_results(
        self,
        paired: bool,
        n_control: int,
        n_candidate: int,
        point: float,
        ci: tuple[float, float],
        boot: Optional[np.ndarray],
        ttest: Optional[dict[str, Any]],
        power: Optional[dict[str, Any]],
    ) -> None:
        if not self.output.print_table:
            return
        kind = "paired" if paired else "unpaired"
        print("")
        print(f"Analysis ({kind})")
        print(f"  metric: {self.metric_key}")
        print(f"  summary: {self.summary.type}")
        print(f"  N_control={n_control}, N_candidate={n_candidate}")
        side = self.bootstrap.side
        print(f"  bootstrapped effect size (candidate - control): {point:.6g}")
        if side == "two-sided":
            print(
                f"  two-sided {1 - self.bootstrap.alpha:.0%} CI: [{ci[0]:.6g}, {ci[1]:.6g}]"
            )
        elif side == "greater":
            print(
                f"  one-sided {1 - self.bootstrap.alpha:.0%} CI (lower): [{ci[0]:.6g}, +inf)"
            )
        else:  # side == "less"
            print(
                f"  one-sided {1 - self.bootstrap.alpha:.0%} CI (upper): (-inf, {ci[1]:.6g}]"
            )
        if boot is not None:
            print("  bootstrap_samples:")
            print(f"    {boot}")
        if ttest is not None:
            print("  t-test:")
            # Loudly print any assumption warnings
            warnings_list = ttest.get("warnings") if isinstance(ttest, dict) else None
            if warnings_list:
                for w in warnings_list:
                    print(f"    WARNING: {w}")
            # Print assumption stats if present
            assumptions = ttest.get("assumptions") if isinstance(ttest, dict) else None
            if isinstance(assumptions, dict):
                for k in sorted(assumptions.keys()):
                    print(f"    {k}: {assumptions[k]}")
            # Print t-test results only when present
            tvals = ttest.get("ttest") if isinstance(ttest, dict) else None
            if isinstance(tvals, dict):
                if "t_stat" in tvals and "p_value" in tvals:
                    print(f"    t_stat: {tvals['t_stat']}")
                    print(f"    p_value: {tvals['p_value']}")
        if power is not None:
            print("  power:")
            for k in sorted(power.keys()):
                print(f"    {k}: {power[k]}")
        print("")

    def _write_csv(
        self,
        path: str,
        summaries: Iterable[tuple[str, str, float] | tuple[str, str, float, float]],
        point: float,
        ci: tuple[float, float],
    ) -> None:
        # Rows: type,label,control?,candidate?,value
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["group", "label", "value_or_control", "candidate_value_optional"]
            )
            for row in summaries:
                if len(row) == 3:
                    grp, label, val = row  # type: ignore[misc]
                    w.writerow([grp, label, f"{val:.9g}", ""])
                else:
                    grp, label, cval, tval = row  # type: ignore[misc]
                    w.writerow([grp, label, f"{cval:.9g}", f"{tval:.9g}"])
            w.writerow(["result", "effect", f"{point:.9g}", ""])
            w.writerow(["result", "ci_low", f"{ci[0]:.9g}", ""])
            w.writerow(["result", "ci_high", f"{ci[1]:.9g}", ""])

    def _power_info(self, n: int, scale: float) -> dict[str, Any]:
        # Simple normal-approx power calc for mean difference
        # target_effect_size is in units of the summary statistic; scale ~ std error proxy
        alpha = self.bootstrap.alpha
        side = self.bootstrap.side
        dist = NormalDist()
        if side == "two-sided":
            z_alpha = dist.inv_cdf(1 - alpha / 2)
        else:
            z_alpha = dist.inv_cdf(1 - alpha)
        z_beta = dist.inv_cdf(self.power.beta)
        # Required N under normal approx: (z_alpha + z_beta)^2 * sigma^2 / d^2
        # We approximate sigma by `scale` argument (caller chooses)
        d = max(1e-12, abs(self.power.target_effect_size))
        required_n = ((z_alpha + z_beta) ** 2) * (scale**2) / (d**2)
        return {
            "approx_required_N": float(required_n),
            "alpha": float(alpha),
            "beta": float(self.power.beta),
            "target_effect_size": float(self.power.target_effect_size),
        }


def compare(
    control_run_ids: list[str] | None = None,
    candidate_run_ids: list[str] | None = None,
    pairs: list[RunPair] | None = None,
    metric_key: str = "overview/reward",
    summary: SummarySpec | None = None,
    fetch: FetchSpec | None = None,
    bootstrap: BootstrapSpec | None = None,
    ttest: TTestSpec | None = None,
    power: PowerSpec | None = None,
    output: OutputSpec | None = None,
) -> CompareTool:
    return CompareTool(
        control_run_ids=control_run_ids,
        candidate_run_ids=candidate_run_ids,
        pairs=pairs,
        metric_key=metric_key,
        summary=summary or SummarySpec(),
        fetch=fetch or FetchSpec(),
        bootstrap=bootstrap or BootstrapSpec(),
        ttest=ttest or TTestSpec(),
        power=power or PowerSpec(),
        output=output or OutputSpec(),
    )
