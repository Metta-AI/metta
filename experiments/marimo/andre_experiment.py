import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Mettabook""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    # Optional: confirm you're set up to connect to the services used in this notebook
    # If the command does not run, run `./install.sh` from your terminal
    # Note: Marimo doesn't support IPython magic commands, use subprocess instead if needed
    import subprocess

    subprocess.run(
        ["metta", "status", "--components=core,system,aws,wandb", "--non-interactive"]
    )
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    from experiments.notebooks.utils.metrics import fetch_metrics
    from experiments.notebooks.utils.monitoring import sky_job_exists, get_sky_jobs_data
    from experiments.notebooks.utils.monitoring_marimo import monitor_training_statuses
    from experiments.notebooks.utils.training import launch_training
    from experiments.notebooks.utils.replays import show_replay

    print("Setup complete!")
    return (
        alt,
        fetch_metrics,
        get_sky_jobs_data,
        launch_training,
        monitor_training_statuses,
        pd,
        show_replay,
        sky_job_exists,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Launch Training
    #### Uncomment the below to launch training
    """
    )
    return


@app.cell
def _(get_sky_jobs_data):
    get_sky_jobs_data()
    return


@app.cell
def _(launch_training, sky_job_exists):
    # Going to launch a job if it doesn't exist
    round = 4
    run_names = []

    job_name = f"andre.baseline.round{round}"
    if not sky_job_exists(job_name):
        print("launching: ", job_name)
        launch_training(
            run_name=job_name,
            wandb_tags=["low_reward"],
            additional_args=["--skip-git-check"],
        )
    run_names.append(job_name)

    job_name = f"andre.move_cardinal.round{round}"
    if not sky_job_exists(job_name):
        print("launching: ", job_name)
        launch_training(
            run_name=job_name,
            wandb_tags=["low_reward"],
            additional_args=[
                "++replay_job.sim.env_overrides.game.actions.move.enabled=false",
                "++replay_job.sim.env_overrides.game.actions.rotate.enabled=false",
                "++replay_job.sim.env_overrides.game.actions.move_cardinal.enabled=true",
                "--skip-git-check",
            ],
        )
    run_names.append(job_name)

    job_name = f"andre.move_8way.round{round}"
    if not sky_job_exists(job_name):
        print("launching: ", job_name)
        launch_training(
            run_name=job_name,
            wandb_tags=["low_reward"],
            additional_args=[
                "++replay_job.sim.env_overrides.game.actions.move.enabled=false",
                "++replay_job.sim.env_overrides.game.actions.rotate.enabled=false",
                "++replay_job.sim.env_overrides.game.actions.move_8way.enabled=true",
                "--skip-git-check",
            ],
        )
    run_names.append(job_name)
    return (run_names,)


@app.cell
def _(mo):
    mo.md(r"""## Monitor Training Jobs""")
    return


@app.cell
def _(monitor_training_statuses, run_names):
    monitor_training_statuses(run_names, show_metrics=["_step", "overview/reward"])
    return


@app.cell
def _(mo):
    mo.md(r"""## Fetch Metrics""")
    return


@app.cell
def _(fetch_metrics, run_names):
    # Option 1: Fetch with sampling (fast, returns 500 data points)
    # metrics_dfs = fetch_metrics(run_names, samples=500)

    # Option 2: Fetch many samples without full scan (good balance)
    # metrics_dfs = fetch_metrics(run_names, samples=10000)

    # Option 3: Fetch only specific metrics (much faster)
    metrics_dfs = fetch_metrics(
        run_names, samples=5000, keys=["overview/reward", "_step", "losses/policy_loss"]
    )

    # Option 4: Fetch specific step range
    # metrics_dfs = fetch_metrics(run_names, samples=None, min_step=1000, max_step=5000)

    # Option 5: Fetch ALL data points (slowest, complete data)
    # metrics_dfs = fetch_metrics(run_names, samples=None)

    return (metrics_dfs,)


@app.cell
def _(metrics_dfs):
    metrics_dfs
    return


@app.cell
def _(mo):
    mo.md(r"""## Analyze Metrics""")
    return


@app.cell
def _(alt, metrics_dfs, pd):
    # Plot overview metrics for all fetched runs
    if not metrics_dfs:
        print("No metrics data available. Please fetch metrics first.")
        display = None
    else:
        print(f"Plotting metrics for {len(metrics_dfs)} runs")

        # Find common metrics across all runs
        all_columns = set()
        for _, _df in metrics_dfs.items():
            all_columns.update(_df.columns)

        columns = ["overview/reward", "losses/explained_variance"]
        plot_cols = []

        for col in all_columns:
            if col not in columns:
                continue
            # Check if this column exists in at least one run with numeric data
            has_numeric_data = False
            for _df in metrics_dfs.values():
                if (
                    col in _df.columns
                    and pd.api.types.is_numeric_dtype(_df[col])
                    and _df[col].nunique() > 1
                ):
                    has_numeric_data = True
                    break
            if has_numeric_data:
                plot_cols.append(col)

        if not plot_cols:
            print("No plottable metrics found")
            charts = None
        else:
            # Prepare data for Altair
            all_data = []
            for run_name, df in metrics_dfs.items():
                for col in plot_cols:
                    if col in df.columns and "_step" in df.columns:
                        temp_df = df[["_step", col]].copy()
                        temp_df["run"] = run_name
                        temp_df["metric"] = col.replace("overview/", "").replace(
                            "_", " "
                        )
                        temp_df["value"] = temp_df[col]
                        temp_df = temp_df[["_step", "run", "metric", "value"]]
                        all_data.append(temp_df)

            # Combine all data
            plot_data = pd.concat(all_data, ignore_index=True)

            # Create individual charts for each metric
            charts = []
            for metric in plot_cols:
                metric_name = metric.replace("overview/", "").replace("_", " ")
                metric_data = plot_data[plot_data["metric"] == metric_name]

                # Create base chart
                base = (
                    alt.Chart(metric_data)
                    .mark_line(strokeWidth=2, opacity=0.8)
                    .encode(
                        x=alt.X("_step:Q", title="Steps", axis=alt.Axis(format="~s")),
                        y=alt.Y("value:Q", title=metric_name),
                        color=alt.Color(
                            "run:N",
                            title="Run",
                            scale=alt.Scale(scheme="category10"),
                            legend=alt.Legend(orient="top", columns=2),
                        ),
                        tooltip=[
                            alt.Tooltip("run:N", title="Run"),
                            alt.Tooltip("_step:Q", title="Step", format=","),
                            alt.Tooltip("value:Q", title=metric_name, format=".4f"),
                        ],
                    )
                    .properties(width=500, height=300, title=metric_name)
                )

                # Add interactive selection
                selection = alt.selection_point(
                    fields=["run"], bind="legend", on="click", clear="dblclick"
                )

                chart = base.add_params(selection).encode(
                    opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
                )

                charts.append(chart)

                display = (
                    alt.vconcat(*charts)
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=12, titleFontSize=14)
                    .configure_title(fontSize=16, anchor="start")
                    .configure_legend(titleFontSize=14, labelFontSize=12)
                )
    display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## View Replays

    Display replay viewer for a specific run:
    """
    )
    return


@app.cell
def _(run_names, show_replay):
    # Show available replays
    # replays = get_available_replays("daveey.lp.16x4.bptt8")

    def show_replays(
        run_names: list[str],
        step: str | int = "last",
        width: int = 1000,
        height: int = 600,
    ) -> None:
        for run_name in run_names:
            show_replay(run_name, step=step, width=width, height=height)

    show_replays(run_names, step="last", width=1000, height=600)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
