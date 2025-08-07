import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Policy Scorecards""")
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    from metta.app_backend.clients.scorecard_client import ScorecardClient
    from metta.common.util.collections import group_by

    client = ScorecardClient()
    return alt, client, group_by, pd


@app.cell
async def _(client, group_by):
    # Load all policies
    policies_response = await client.get_policies()
    all_policies = policies_response.policies

    # Separate training runs and run-free policies
    policy_groups = group_by(all_policies, key_fn=lambda x: x.type)
    training_run_policies = policy_groups.get("training_run", [])
    run_free_policies = policy_groups.get("policy", [])

    print(
        f"Fetched {len(all_policies)} policies:\n"
        f"  - training runs: {len(training_run_policies)}\n"
        f"  - run-free policies: {len(run_free_policies)}"
    )
    return run_free_policies, training_run_policies


@app.cell
def _(mo):
    mo.md("""## 1. Select Training Runs""")
    return


@app.cell
def _(mo, run_free_policies, training_run_policies):
    # Filter to show only navigation-related runs
    training_run_selector = mo.ui.multiselect(
        options={p.name: p.id for p in training_run_policies + run_free_policies},
        value=[],
        full_width=True,
        label="Select runs:",
        max_selections=30,
    )

    training_run_selector
    return (training_run_selector,)


@app.cell
def _(mo):
    mo.md(r"""## 2. Select Evals""")
    return


@app.cell
async def _(client, group_by, mo, training_run_selector):
    if training_run_selector.value:
        available_evals = await client.get_eval_names(
            training_run_ids=training_run_selector.value, run_free_policy_ids=[]
        )
        eval_by_category = group_by(available_evals, key_fn=lambda e: e.split("/")[0])
    else:
        available_evals = []
        eval_by_category = {}

    # Create multiselect for each category
    category_selectors = {}
    ui_elements = []

    for cat_key in sorted(eval_by_category.keys()):
        cat_evals = eval_by_category[cat_key]
        # Format eval names nicely for display

        cat_multiselect = mo.ui.multiselect(
            options=cat_evals,
            value=cat_evals,
            label=f"{cat_key.replace('_', ' ').title()}",
        )
        category_selectors[cat_key] = cat_multiselect
        ui_elements.append(cat_multiselect)

    # Display all category selectors vertically for consistency
    display = (
        mo.vstack(
            [
                mo.md("*Select which evaluations to include in the scorecard:*"),
                mo.vstack(ui_elements, gap=1),  # All selectors in vertical layout
            ]
        )
        if ui_elements
        else mo.md("Select training runs to continue")
    )

    display
    return (category_selectors,)


@app.cell
def _(mo):
    mo.md(r"""## 3. Configure Scorecard""")
    return


@app.cell
async def _(category_selectors, client, training_run_selector):
    selected_eval_names = []
    for cat_name, cat_selector in category_selectors.items():
        selected_eval_names.extend(cat_selector.value)

    # Load available metrics
    if training_run_selector.value and selected_eval_names:
        available_metrics = await client.get_available_metrics(
            training_run_ids=training_run_selector.value,
            run_free_policy_ids=[],
            eval_names=selected_eval_names,
        )
    else:
        available_metrics = []
    return available_metrics, selected_eval_names


@app.cell
def _(available_metrics, mo):
    metric_selector = mo.ui.dropdown(
        options=sorted(available_metrics + ["reward"]),
        value="reward",
        label="Metric:",
    )
    policy_selector_selector = mo.ui.radio(
        options=["best", "latest"],
        value="best",
        label="Policy version:",
    )

    mo.vstack([metric_selector, policy_selector_selector], gap=2)
    return metric_selector, policy_selector_selector


@app.cell
def _(metric_selector, mo, selected_eval_names, training_run_selector):
    can_generate = (
        training_run_selector.value and selected_eval_names and metric_selector.value
    )
    generate_button = mo.ui.button(
        label="Generate Scorecard",
        disabled=not can_generate,
        kind="success" if can_generate else "neutral",
        value=0,
        on_click=lambda value: value + 1,
    )

    generate_button
    return (generate_button,)


@app.cell
async def _(
    client,
    generate_button,
    metric_selector,
    mo,
    policy_selector_selector,
    selected_eval_names,
    training_run_selector,
):
    scorecard_data = None
    if generate_button.value:
        with mo.status.spinner("Generating scorecard..."):
            scorecard_data = await client.generate_scorecard(
                training_run_ids=training_run_selector.value,
                run_free_policy_ids=[],
                eval_names=selected_eval_names,
                metric=metric_selector.value,
                policy_selector=policy_selector_selector.value,
            )
    return (scorecard_data,)


@app.cell
def _(pd, scorecard_data):
    def _get_table_df():
        sorted_policies = sorted(
            (scorecard_data and scorecard_data.policyNames) or [],
            key=lambda p: scorecard_data.policyAverageScores.get(p, 0),
            reverse=True,
        )
        if not (scorecard_data and sorted_policies):
            return pd.DataFrame({"Policy": [], "Overall Score": []})
        rows = []
        for policy in sorted_policies:
            row = {
                "Policy": policy.split(":v")[0][:40] if ":v" in policy else policy[:40],
                "Overall Score": round(
                    scorecard_data.policyAverageScores.get(policy, 0), 1
                ),
                "_original_policy": policy,
            }

            # Add evaluation scores
            for eval_name in scorecard_data.evalNames:
                cell_data = scorecard_data.cells.get(policy, {}).get(eval_name)
                eval_display = (
                    eval_name.split("/", 1)[1].replace("_", " ").title()
                    if "/" in eval_name
                    else eval_name.replace("_", " ").title()
                )
                if not cell_data:
                    row[eval_display] = None
                else:
                    value = (
                        cell_data.value if hasattr(cell_data, "value") else cell_data
                    )
                    if isinstance(value, (int, float)):
                        row[eval_display] = round(value, 2)
                    else:
                        row[eval_display] = value
            rows.append(row)

        visualization_df = pd.DataFrame(rows)
        return visualization_df.drop(columns=["_original_policy"])

    table_df = _get_table_df()
    return (table_df,)


@app.cell
def _(alt, metric_selector, mo, table_df):
    def _get_heatmap(table_df):
        if len(table_df) > 0:
            # Heatmap view
            eval_cols = [
                col
                for col in table_df.columns
                if col not in ["Policy", "Overall Score"]
            ]

            if eval_cols:  # Only create heatmap if we have evaluation columns
                # Melt data for Altair
                df_melted = table_df.melt(
                    id_vars=["Policy", "Overall Score"],
                    var_name="Evaluation",
                    value_name="Score",
                ).dropna(subset=["Score"])

                # Create heatmap
                heatmap = (
                    alt.Chart(df_melted)
                    .mark_rect(stroke="white", strokeWidth=1)
                    .encode(
                        x=alt.X(
                            "Evaluation:N",
                            title="Evaluation",
                            axis=alt.Axis(labelAngle=-45, labelLimit=200),
                        ),
                        y=alt.Y(
                            "Policy:N",
                            title="Policy",
                            sort=alt.SortField("Overall Score", order="descending"),
                        ),
                        color=alt.Color(
                            "Score:Q",
                            scale=alt.Scale(scheme="redyellowgreen"),
                            title="Score",
                        ),
                        tooltip=[
                            alt.Tooltip("Policy:N", title="Policy"),
                            alt.Tooltip("Evaluation:N", title="Evaluation"),
                            alt.Tooltip("Score:Q", title="Score", format=".1f"),
                        ],
                    )
                    .properties(
                        width=800,
                        height=max(400, len(table_df) * 25),
                        title=f"Policy Performance Heatmap - {metric_selector.value.upper()}",
                    )
                )

                return mo.ui.altair_chart(heatmap)
            else:
                return mo.md("No evaluation data to display in heatmap")
        else:
            return mo.md("No data to display")

    mo.vstack([_get_heatmap(table_df), table_df], align="stretch")
    return


if __name__ == "__main__":
    app.run()
