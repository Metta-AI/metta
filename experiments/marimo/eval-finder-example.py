import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Eval Finder Widget Example""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Widget Features

    The Eval Finder Widget provides:

    - **🪟 Views**: Tree view, list view, category view
    - **🔍 Search**: Filter by name
    - **🎯 Multi-select**: Choose as many as you like!

    ## Integration with Scorecard

    Selected evaluations can be used directly with the scorecard widget for evaluation (I'll show you; scroll down...)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Setup

    The Eval Finder Widget helps you discover and select evaluations for your policies.
    It provides filtering by agent type, difficulty, and category, plus shows prerequisite relationships.
    """
    )
    return


@app.cell
def _():
    # Import the eval finder widget
    from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget import (
        EvalFinderWidget,
    )
    from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget.util import (
        create_demo_eval_finder_widget,
        fetch_eval_data_for_policies,
    )
    from metta.app_backend.clients.scorecard_client import ScorecardClient

    # Comment one of these out, uncomment the other.
    # client = ScorecardClient()  # production: https://api.observatory.softmax-research.net
    client = ScorecardClient(backend_url="http://localhost:8000")  # development

    print("🎯 Eval Finder Widget imported successfully!")
    return (
        EvalFinderWidget,
        client,
        create_demo_eval_finder_widget,
        fetch_eval_data_for_policies,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## Demo Widget

    Let's start with a demo widget that has sample data:
    """
    )
    return


@app.cell
def _(create_demo_eval_finder_widget, mo):
    # Create demo widget with sample data
    demo_widget = mo.ui.anywidget(create_demo_eval_finder_widget())
    demo_widget
    return (demo_widget,)


@app.cell
def _(demo_widget, mo):
    # INFO: RUN THIS CELL MANUALLY
    # Show selected evaluations from demo widget
    selected = demo_widget.selected_evals

    mo.vstack(
        [
            mo.md(f"## Selected evals (demo data): {len(selected)} evaluations"),
            mo.md(f"**Current Selection:** {len(selected)} evaluation(s)"),
            mo.md(f"Selected: {', '.join(selected) if selected else 'None'}")
            if selected
            else mo.md("### No evaluations selected"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Get Available Policies/Training Runs

    First, let's see what policies and training runs are available:
    """
    )
    return


@app.cell
async def _(client):
    async def _():
        # Get available policies and training runs
        try:
            print("🔗 Testing connection to backend...")
            policies_response = await client.get_policies()
            all_policies = policies_response.policies

            # Separate training runs and run-free policies
            training_run_policies = [
                p for p in all_policies if p.type == "training_run"
            ]
            run_free_policies = [p for p in all_policies if p.type == "policy"]

            print(
                f"📊 Found {len(training_run_policies)} training runs and {len(run_free_policies)} standalone policies"
            )

            # Show a few examples
            if training_run_policies:
                print("🏃 Sample training runs:")
                for p in training_run_policies[:5]:
                    print(f"  - {p.name} ({p.id[:8]}...)")
                if len(training_run_policies) > 5:
                    print("  - ...")

            if run_free_policies:
                print("🤖 Sample standalone policies:")
                for p in run_free_policies[:5]:
                    print(f"  - {p.name} ({p.id[:8]}...)")
                if len(run_free_policies) > 5:
                    print("  - ...")

            return training_run_policies, run_free_policies

        except Exception as e:
            import traceback

            print(f"⚠️ Could not fetch policies: {e}")
            print(f"Exception type: {type(e).__name__}")
            print("Traceback:")
            traceback.print_exc()
            return [], []

    training_run_policies, run_free_policies = await _()
    return run_free_policies, training_run_policies


@app.cell
def _(mo, run_free_policies, training_run_policies):
    md = mo.md("""
    ## Select Policies for Context-Aware Eval Discovery
    Choose policies/runs to get contextual eval recommendations:
    """)

    # Policy selection UI
    policy_selector = mo.ui.multiselect(
        options={
            p.name: p.id for p in (training_run_policies + run_free_policies)[:1000]
        },  # Limit for UI performance
        value=[],
        label="Select policies/training runs:",
        max_selections=64,
    )

    mo.vstack(
        [
            md,
            policy_selector,
        ]
    )
    return (policy_selector,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Policy-Aware Eval Finder Widget

    This widget shows evaluations contextually based on your selected policies:
    """
    )
    return


@app.cell
def _(
    EvalFinderWidget,
    client,
    create_demo_eval_finder_widget,
    fetch_eval_data_for_policies,
    mo,
    policy_selector,
    training_run_policies,
):
    # Make a widget and wrap it
    eval_finder = EvalFinderWidget()
    mo_eval_finder = mo.ui.anywidget(eval_finder)

    try:
        # Determine which are training runs vs standalone policies
        selected_training_runs = []
        _selected_policies = []

        for policy_id in policy_selector.value:
            # Check if it's a training run
            is_training_run = any(p.id == policy_id for p in training_run_policies)
            if is_training_run:
                selected_training_runs.append(policy_id)
            else:
                _selected_policies.append(policy_id)

        print(
            f"🎯 Fetching evals for {len(selected_training_runs)} training runs and {len(_selected_policies)} policies"
        )

        # Fetch policy-aware eval data
        eval_data = fetch_eval_data_for_policies(
            training_run_ids=selected_training_runs,
            run_free_policy_ids=_selected_policies,
            category_filter=[],
            client=client,
        )

        # Set the data on the raw widget
        eval_finder.set_eval_data(
            evaluations=eval_data["evaluations"],
            # categories=eval_data["categories"], # you can leave this unset to fetch all categories
        )

        print(f"📊 Loaded {len(eval_data['evaluations'])} evaluations")

    except Exception as e:
        print(f"⚠️ Could not fetch live data: {e}")
        print("Using demo data instead...")
        # Fallback to demo data if backend not available
        eval_finder = create_demo_eval_finder_widget()
        mo_eval_finder = mo.ui.anywidget(eval_finder)

    mo_eval_finder
    return eval_finder, mo_eval_finder


@app.cell
def _(mo, mo_eval_finder):
    # The wrapped widget's value is a dict of all traits, so we need to extract selected_evals
    selection = mo_eval_finder.value["selected_evals"]

    def _():
        # Show different content based on selection
        if not selection:
            return mo.callout(
                mo.md("## 👆 Select some evaluations above to see them here!"),
                kind="danger",
            )

        return mo.vstack(
            [
                mo.md("## ✨ Auto-Reactive Selection ✨"),
                mo.md(f"**Selection:** {len(selection)} evaluations"),
                mo.md(f"Selected: {', '.join(selection)}"),
                mo.callout(
                    mo.md(
                        "*This cell automatically updates when you change selections!*"
                    ),
                    kind="success",
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Filtering Examples

    You can programmatically control the widget filters:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 🎯 Ready for Scorecard

    You can now use selected evaluations with the scorecard widget:
    """
    )
    return


@app.cell
async def _(
    client,
    mo_eval_finder,
    policy_selector,
    run_free_policies,
    training_run_policies,
):
    from experiments.notebooks.utils.scorecard_widget.scorecard_widget.util import (
        fetch_real_scorecard_data,
    )

    # Get the selected policy IDs from the selector
    selected_policy_ids = policy_selector.value

    # Find the names of the selected policies
    all_policies = training_run_policies + run_free_policies
    selected_policy_names = [
        policy.name for policy in all_policies if policy.id in selected_policy_ids
    ]

    # Extract selected_evals from the widget's value dict
    selected_evals = mo_eval_finder.value["selected_evals"]

    print(f"🔍 Selected policy IDs: {selected_policy_ids}")
    print(f"🔍 Selected policy names: {selected_policy_names}")
    print(f"🔍 Selected evaluations: {selected_evals}")

    scorecard_widget = None
    if selected_evals:  # Check the actual list, not the state object
        try:
            # Generate scorecard using the selected evaluations and policies
            scorecard_widget = await fetch_real_scorecard_data(
                client=client,
                restrict_to_policy_names=selected_policy_names,  # Only selected policies!
                restrict_to_metrics=["reward"],  # Focus on reward metric
                restrict_to_eval_names=selected_evals,  # Use selected evals from widget
                policy_selector="best",
                max_policies=10,
            )

            print("📊 Generated scorecard")
        except Exception as e:
            print(f"⚠️ Could not generate scorecard: {e}")
    else:
        print("No evaluations selected - select some evaluations first")

    scorecard_widget
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Callback Examples

    You can register callbacks to respond to selection changes:
    """
    )
    return


@app.cell
def _(eval_finder):
    # Register callbacks for demo
    def on_selection_changed(event):
        selected_evals = event.get("selected_evals", [])
        if selected_evals:
            print(
                f"   Selected: {', '.join(selected_evals[:3])}{'...' if len(selected_evals) > 3 else ''}"
            )

    eval_finder.on_selection_changed(on_selection_changed)

    print("✅ Selection callback registered")
    return


if __name__ == "__main__":
    app.run()
