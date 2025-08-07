import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Mettabook

    ## Setup
    """
    )
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
    import pandas as pd
    import altair as alt

    from experiments.notebooks.utils.metrics import fetch_metrics
    from experiments.notebooks.utils.monitoring import monitor_training_statuses
    from experiments.notebooks.utils.replays import show_replay

    # Configure Altair theme for better visuals
    # You can try 'default', 'dark', 'fivethirtyeight', 'ggplot2', 'latimes', 'powerbi', 'quartz', 'urbaninstitute', 'vox'
    # alt.themes.enable('default')  # Clean default theme

    print("Setup complete!")
    return alt, fetch_metrics, monitor_training_statuses, pd, show_replay


@app.cell
def _(mo):
    mo.md(r"""## Launch Training""")
    return


@app.cell
def _():
    # Example: Launch training

    # new_run_name = f"{os.environ.get('USER')}.training-run.{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # print(f"Launching training with run name: {new_run_name}...")

    # # # View `launch_training` function for all options
    # result = launch_training(
    #     run_name=new_run_name,
    #     curriculum="env/mettagrid/arena/basic",
    #     wandb_tags=[f"{os.environ.get('USER')}-arena-experiment"],
    #     additional_args=["--skip-git-check"],
    # )
    return


@app.cell
def _(mo):
    mo.md(r"""## Monitor Training Jobs""")
    return


@app.cell
def _(monitor_training_statuses):
    # Monitor Training
    run_names = [
        "daveey.navigation.low_reward.baseline.2",
        "daveey.navigation.low_reward.baseline.07-18",
    ]

    # Optional: instead, find all runs that meet some criteria
    # runs = find_training_runs(
    #     # wandb_tags=["low_reward"],
    #     # state="finished",
    #     author=os.getenv("USER"),
    #     limit=5,
    # )
    # run_names = [run.name for run in runs]  # Extract run names from Run objects

    monitor_training_statuses(run_names, show_metrics=["_step", "overview/reward"])
    return (run_names,)


@app.cell
def _(mo):
    mo.md(r"""## Fetch Metrics""")
    return


@app.cell
def _(fetch_metrics, run_names):
    metrics_dfs = fetch_metrics(run_names, samples=500)
    return (metrics_dfs,)


@app.cell
def _(mo):
    mo.md(r"""## Analyze Metrics""")
    return


@app.cell
def _(alt, metrics_dfs, pd):
    # Plot overview metrics for all fetched runs
    if not metrics_dfs:
        print("No metrics data available. Please fetch metrics first.")
        charts = None
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

            # Combine charts vertically if multiple, otherwise just configure the single chart
            if len(charts) > 1:
                charts = (
                    alt.vconcat(*charts)
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=12, titleFontSize=14)
                    .configure_title(fontSize=16, anchor="start")
                    .configure_legend(titleFontSize=14, labelFontSize=12)
                )
            elif charts:
                charts = (
                    charts[0]
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=12, titleFontSize=14)
                    .configure_title(fontSize=16, anchor="start")
                    .configure_legend(titleFontSize=14, labelFontSize=12)
                )

    return charts


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
def _(show_replay):
    # Show available replays
    # replays = get_available_replays("daveey.lp.16x4.bptt8")

    # Show the last replay for a run
    show_replay("daveey.lp.16x4.bptt8", step="last", width=1000, height=600)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Scorecard Widget

    A marimo-native implementation of the policy evaluation scorecard.
    """
    )
    return


@app.cell
async def _():
    from metta.app_backend.clients.scorecard_client import ScorecardClient

    # Initialize the scorecard client
    scorecard_client = ScorecardClient()
    return ScorecardClient, scorecard_client


@app.cell
def _(mo):
    mo.md(r"""### Frontend-Style Policy Scorecard Dashboard""")
    return


@app.cell
async def _(scorecard_client):
    # Step 1: Load all policies (mimicking Dashboard.tsx useEffect)
    policies_response = await scorecard_client.get_policies()
    all_policies = policies_response.policies

    # Separate training runs and run-free policies
    training_run_policies = [p for p in all_policies if p.type == "training_run"]
    run_free_policies = [p for p in all_policies if p.type == "policy"]

    return all_policies, policies_response, run_free_policies, training_run_policies


@app.cell
def _(mo, training_run_policies):
    mo.md(f"""
    **Available Policies**: {len(training_run_policies)} training runs loaded
    
    Select training runs and evaluations to generate a scorecard.
    """)
    return


@app.cell
def _(mo, training_run_policies):
    # Create searchable multi-select for training runs
    # Filter to show only navigation-related runs for this example
    navigation_runs = [
        p
        for p in training_run_policies
        if "navigation" in p.name.lower() or "nav" in p.name.lower()
    ][:20]  # Limit to 20 for display

    training_run_selector = mo.ui.multiselect(
        options={p.id: p.name for p in navigation_runs},
        value=[],
        label="Select Training Runs:",
    )

    mo.vstack(
        [
            mo.md("#### 1. Select Training Runs"),
            training_run_selector,
            mo.md(f"*Showing {len(navigation_runs)} navigation-related runs*"),
        ]
    )
    return navigation_runs, training_run_selector


@app.cell
async def _(scorecard_client, training_run_selector):
    # Step 2: Load evaluation names when training runs are selected
    selected_training_run_ids = list(training_run_selector.value)

    if selected_training_run_ids:
        eval_names_set = await scorecard_client.get_eval_names(
            training_run_ids=selected_training_run_ids, run_free_policy_ids=[]
        )
        available_evals = list(eval_names_set)
    else:
        available_evals = []

    return available_evals, eval_names_set, selected_training_run_ids


@app.cell
def _(available_evals, mo, selected_training_run_ids):
    if selected_training_run_ids and available_evals:
        # Group evaluations by category
        eval_by_category = {}
        for eval_name in available_evals:
            if "/" in eval_name:
                category, name = eval_name.split("/", 1)
            else:
                category = "misc"
            if category not in eval_by_category:
                eval_by_category[category] = []
            eval_by_category[category].append(eval_name)

        # Create checkboxes for each category
        eval_selectors = {}
        for category, evals in sorted(eval_by_category.items()):
            eval_selectors[category] = mo.ui.array(
                [
                    mo.ui.checkbox(value=True, label=eval_name)
                    for eval_name in sorted(evals)
                ]
            )

        mo.vstack(
            [
                mo.md("#### 2. Select Evaluations"),
                mo.vstack(
                    [
                        mo.vstack([mo.md(f"**{category}**"), eval_selectors[category]])
                        for category in sorted(eval_selectors.keys())
                    ]
                ),
            ]
        )
        return eval_by_category, eval_selectors
    else:
        mo.md("*Select training runs to see available evaluations*")
        return {}, {}


@app.cell
def _(available_evals, eval_selectors):
    # Get selected evaluations
    selected_eval_names = []
    if "eval_selectors" in locals() and eval_selectors:
        for category, selector_array in eval_selectors.items():
            for i, checkbox in enumerate(selector_array):
                if checkbox.value:
                    # Get the eval name from the category's eval list
                    category_evals = sorted(
                        [e for e in available_evals if e.startswith(f"{category}/")]
                    )
                    if i < len(category_evals):
                        selected_eval_names.append(category_evals[i])

    return (selected_eval_names,)


@app.cell
async def _(scorecard_client, selected_eval_names, selected_training_run_ids):
    # Step 3: Load available metrics
    if selected_training_run_ids and selected_eval_names:
        available_metrics = await scorecard_client.get_available_metrics(
            training_run_ids=selected_training_run_ids,
            run_free_policy_ids=[],
            eval_names=selected_eval_names,
        )
    else:
        available_metrics = []

    return (available_metrics,)


@app.cell
def _(available_metrics, mo, selected_eval_names, selected_training_run_ids):
    if selected_training_run_ids and selected_eval_names and available_metrics:
        # Default to 'reward' if available, otherwise first metric
        default_metric = (
            "reward"
            if "reward" in available_metrics
            else (available_metrics[0] if available_metrics else "")
        )

        metric_selector = mo.ui.dropdown(
            options=sorted(available_metrics),
            value=default_metric,
            label="Select Metric:",
        )

        # Policy selector (best vs latest)
        policy_selector = mo.ui.radio(
            options=["best", "latest"], value="best", label="Policy Selection Strategy:"
        )

        mo.vstack(
            [
                mo.md("#### 3. Configure Scorecard"),
                mo.hstack([metric_selector, policy_selector], gap=2),
            ]
        )
    else:
        mo.md("*Select evaluations to see available metrics*")
        metric_selector = None
        policy_selector = None

    return default_metric, metric_selector, policy_selector


@app.cell
def _(
    mo, policy_selector, selected_eval_names, selected_training_run_ids, metric_selector
):
    # Generate scorecard button
    can_generate = (
        selected_training_run_ids
        and selected_eval_names
        and metric_selector
        and metric_selector.value
    )

    generate_button = mo.ui.button(
        label="Generate Scorecard",
        disabled=not can_generate,
        kind="success" if can_generate else "neutral",
    )

    if generate_button.value:
        mo.md(f"""
        **Ready to generate scorecard:**
        - {len(selected_training_run_ids)} training runs
        - {len(selected_eval_names)} evaluations  
        - Metric: {metric_selector.value if metric_selector else "None"}
        - Strategy: {policy_selector.value if policy_selector else "best"}
        """)
    else:
        generate_button

    return can_generate, generate_button


@app.cell
async def _(
    can_generate,
    generate_button,
    mo,
    scorecard_client,
    selected_eval_names,
    selected_training_run_ids,
    metric_selector,
    policy_selector,
):
    if generate_button.value and can_generate:
        # Generate the scorecard (mimicking Dashboard.tsx generateScorecard)
        mo.md("🔄 Generating scorecard...")

        scorecard_data = await scorecard_client.generate_scorecard(
            training_run_ids=selected_training_run_ids,
            run_free_policy_ids=[],
            eval_names=selected_eval_names,
            metric=metric_selector.value,
            training_run_policy_selector=policy_selector.value,
        )

        mo.md("✅ Scorecard generated!")
    else:
        scorecard_data = None

    return (scorecard_data,)


@app.cell
def _(metric_selector, mo, scorecard_data):
    def create_marimo_scorecard(data, metric, num_policies=20):
        """Create a marimo-native scorecard visualization."""
        if not data:
            return mo.Html("<p>No scorecard data available</p>")

        # ScorecardData has these attributes: cells, evalNames, policyNames, policyAverageScores
        cells = data.cells
        eval_names = data.evalNames
        policy_names = data.policyNames
        policy_averages = data.policyAverageScores

        # Sort policies by average score (worst to best, for display)
        sorted_policies = sorted(policy_names, key=lambda p: policy_averages.get(p, 0))[
            -num_policies:
        ]

        # Group evaluations by category
        eval_by_category = {}
        for eval_name in eval_names:
            if "/" in eval_name:
                category, name = eval_name.split("/", 1)
            else:
                category = "misc"
                name = eval_name
            if category not in eval_by_category:
                eval_by_category[category] = []
            eval_by_category[category].append((name, eval_name))

        # Build HTML table
        html = f"""
        <style>
            .scorecard-container {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 20px 0;
                overflow-x: auto;
            }}
            .scorecard-table {{
                border-collapse: collapse;
                font-size: 14px;
                min-width: 600px;
            }}
            .scorecard-table th {{
                padding: 8px;
                text-align: center;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .scorecard-table td {{
                padding: 8px;
                text-align: center;
                border: 1px solid #dee2e6;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .scorecard-table td:first-child {{
                text-align: left;
                font-weight: 500;
                background-color: #f8f9fa;
                position: sticky;
                left: 0;
                z-index: 5;
            }}
            .scorecard-table tr:hover td {{
                background-color: #f0f0f0;
            }}
            .scorecard-cell {{
                position: relative;
                min-width: 60px;
            }}
            .category-header {{
                background-color: #e9ecef !important;
                font-size: 12px;
                text-transform: uppercase;
            }}
            .metric-selector {{
                margin: 10px 0;
                padding: 8px 12px;
                font-size: 14px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }}
            .scorecard-title {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 10px;
                color: #212529;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
        <div class="scorecard-container">
            <div class="scorecard-title">Policy Scorecard - Metric: {metric}</div>
            <table class="scorecard-table">
                <thead>
                    <tr>
                        <th>Policy</th>
                        <th>Overall</th>
        """

        # Add category headers
        for category in sorted(eval_by_category.keys()):
            colspan = len(eval_by_category[category])
            html += f'<th colspan="{colspan}" class="category-header">{category}</th>'
        html += "</tr><tr><th></th><th></th>"

        # Add evaluation name headers
        for category in sorted(eval_by_category.keys()):
            for name, _ in eval_by_category[category]:
                html += f"<th>{name}</th>"
        html += "</tr></thead><tbody>"

        # Add data rows
        for policy in reversed(sorted_policies):  # Best policies at top
            policy_avg = policy_averages.get(policy, 0)

            # Create policy link if it contains version info
            if ":v" in policy:
                policy_display = f'<a href="https://wandb.ai/softmax-ai/metta/runs/{policy.split(":v")[0]}" target="_blank">{policy}</a>'
            else:
                policy_display = policy

            html += f"<tr><td>{policy_display}</td>"

            # Overall score with color
            color = get_score_color(policy_avg, 0, 100)
            html += f'<td class="scorecard-cell" style="background-color: {color};">{policy_avg:.1f}</td>'

            # Individual evaluation scores
            for category in sorted(eval_by_category.keys()):
                for _, eval_name in eval_by_category[category]:
                    # In ScorecardData, cells[policy][eval_name] directly contains the value
                    cell_data = cells.get(policy, {}).get(eval_name)
                    if cell_data:
                        value = cell_data.value
                        color = get_score_color(value, 0, 100)
                        html += f'<td class="scorecard-cell" style="background-color: {color};" title="{eval_name}: {value:.2f}">{value:.1f}</td>'
                    else:
                        html += '<td class="scorecard-cell" style="background-color: #f8f9fa;">-</td>'

            html += "</tr>"

        html += "</tbody></table></div>"

        return mo.Html(html)

    def get_score_color(value, min_val=0, max_val=100):
        """Get color for score visualization."""
        if value is None:
            return "#f8f9fa"

        # Normalize to 0-1
        norm_value = (value - min_val) / (max_val - min_val)
        norm_value = max(0, min(1, norm_value))

        # Color gradient from red to yellow to green
        if norm_value < 0.5:
            # Red to yellow
            r = 255
            g = int(255 * norm_value * 2)
            b = 0
        else:
            # Yellow to green
            r = int(255 * (1 - (norm_value - 0.5) * 2))
            g = 255
            b = 0

        return f"rgba({r}, {g}, {b}, 0.3)"

    # Display the scorecard if data is available
    if scorecard_data and metric_selector:
        current_metric = metric_selector.value if metric_selector else "reward"
        scorecard_display = create_marimo_scorecard(
            scorecard_data, current_metric, num_policies=15
        )
        return scorecard_display
    else:
        return mo.Html("<p>Generate a scorecard to see results</p>")


if __name__ == "__main__":
    app.run()
