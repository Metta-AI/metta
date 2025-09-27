import marimo

__generated_with = "0.14.17"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell
def _():
    from experiments.notebooks.utils.policy_selector_widget.policy_selector_widget import (
        create_policy_selector_widget,
    )

    widget = create_policy_selector_widget()
    try:
        test_policies = [
            {
                "id": "test-1",
                "type": "training_run",
                "name": "Test Training Run",
                "user_id": "test",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["test"],
            }
        ]
        widget.set_policy_data(test_policies)
        widget
    except Exception as e:
        f"Error: {e}"

    widget
    return (create_policy_selector_widget,)


@app.cell
def _():
    mo.md(r"""## Let's try with some real data from Metta's HTTP API""")
    return


@app.cell
def _():
    from softmax.orchestrator.clients.scorecard_client import ScorecardClient

    client = ScorecardClient()  # production data
    # client = ScorecardClient("http://localhost:8000")  # development mode
    return (client,)


@app.cell
def _(client, create_policy_selector_widget):
    live_widget = mo.ui.anywidget(create_policy_selector_widget(client=client))

    live_widget
    return (live_widget,)


@app.cell
def _(live_widget):
    # Access the widget's value to trigger reactivity in Marimo
    selected_policies = live_widget.selected_policies

    mds = [mo.md(f"## You selected {len(selected_policies)} policies:")]
    mds.append(
        mo.md("\n".join([f"  - {policy_id}" for policy_id in selected_policies]))
    )

    mo.vstack(mds)
    return


@app.cell
async def _(client, live_widget):
    # Access the widget's value to trigger reactivity in Marimo
    from experiments.notebooks.utils.scorecard_widget.scorecard_widget.ScorecardWidget import (
        ScorecardWidget,
    )

    policies_for_scorecard = live_widget.selected_policies
    scorecard_widget = ScorecardWidget(client=client)

    await scorecard_widget.fetch_real_scorecard_data(
        restrict_to_metrics=["heart.get", "reward"],
        restrict_to_policy_ids=policies_for_scorecard,
        policy_selector="latest",
        max_policies=20,
    )

    scorecard_widget
    return


if __name__ == "__main__":
    app.run()
