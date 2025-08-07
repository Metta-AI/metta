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
    # Policy Scorecard Dashboard

    Interactive dashboard for evaluating and comparing policy performance across different metrics and evaluations.
    """
    )
    return


@app.cell
def _():
    from metta.app_backend.clients.scorecard_client import ScorecardClient
    from experiments.marimo.utils import (
        create_marimo_scorecard,
    )

    client = ScorecardClient()
    return client, create_marimo_scorecard


@app.cell
async def _(client):
    # Load all policies
    policies_response = await client.get_policies()
    all_policies = policies_response.policies

    # Separate training runs and run-free policies
    training_run_policies = [p for p in all_policies if p.type == "training_run"]
    run_free_policies = [p for p in all_policies if p.type == "policy"]
    print(
        f"Fetched {len(training_run_policies)} training run policies and {len(run_free_policies)} run-free policies"
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
async def _(client, mo, training_run_selector):
    from metta.common.util.collections import group_by

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
    mo.md(r"""# 3. Configure Scorecard""")
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

    policy_selector = mo.ui.radio(
        options=["best", "latest"],
        value="best",
        label="Policy version:",
    )

    num_policies_slider = mo.ui.slider(
        start=5,
        stop=30,
        value=15,
        step=5,
        label="Number of policies:",
    )

    # Display all selectors vertically for better alignment
    mo.vstack([metric_selector, policy_selector, num_policies_slider], gap=2)
    return metric_selector, num_policies_slider, policy_selector


@app.cell
def _(
    metric_selector,
    mo,
    num_policies_slider,
    selected_eval_names,
    training_run_selector,
):
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

    mo.vstack(
        [
            generate_button,
            mo.md(f"""
        **Selected:**

        - {len(training_run_selector.value)} training runs
        - {len(selected_eval_names)} evals
        - Metric: {metric_selector.value}
        - Show top {num_policies_slider.value} policies
        """)
            if can_generate
            else mo.md("Select runs and evals to generate a scorecard"),
        ]
    )
    return (generate_button,)


@app.cell
async def _(
    client,
    generate_button,
    metric_selector,
    mo,
    policy_selector,
    selected_eval_names,
    training_run_selector,
):
    if generate_button.value:
        with mo.status.spinner("Generating scorecard..."):
            scorecard_data = await client.generate_scorecard(
                training_run_ids=training_run_selector.value,
                run_free_policy_ids=[],
                eval_names=selected_eval_names,
                metric=metric_selector.value,
                policy_selector=policy_selector.value,
            )

    else:
        scorecard_data = None
        print("Have not opted to generate yet")
    return (scorecard_data,)


@app.cell
def _(
    create_marimo_scorecard,
    metric_selector,
    mo,
    num_policies_slider,
    scorecard_data,
):
    mo.vstack(
        [
            mo.md("## Scorecard Results"),
            create_marimo_scorecard(
                scorecard_data,
                metric_selector.value,
                num_policies=num_policies_slider.value,
            ),
        ]
    ) if scorecard_data else mo.md("")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
