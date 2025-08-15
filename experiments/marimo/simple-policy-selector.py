import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Simple Policy Selector Test""")
    return


@app.cell
def _():
    from experiments.notebooks.utils.policy_selector_widget import (
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
    from metta.app_backend.clients.scorecard_client import ScorecardClient

    # client = ScorecardClient()  # production data
    client = ScorecardClient("http://localhost:8000")  # development mode
    return (client,)


@app.cell
def _(client, create_policy_selector_widget):
    live_widget = create_policy_selector_widget(client=client)

    live_widget
    return


if __name__ == "__main__":
    app.run()
