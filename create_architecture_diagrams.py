#!/usr/bin/env python3
"""
Generate architecture diagrams for the Metta refactoring proposal.
This script creates visual comparisons between current and proposed architectures.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch


def create_current_architecture_diagram():
    """Create a diagram showing the current monolithic architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Main LayerBase monolith
    main_box = FancyBboxPatch(
        (2, 2), 8, 4, boxstyle="round,pad=0.1", facecolor="lightcoral", edgecolor="darkred", linewidth=2
    )
    ax.add_patch(main_box)

    # LayerBase title
    ax.text(6, 5.5, "LayerBase", fontsize=16, fontweight="bold", ha="center", va="center")
    ax.text(6, 5, "(327 lines)", fontsize=10, ha="center", va="center", style="italic")

    # Responsibilities inside LayerBase
    responsibilities = [
        "• Component Lifecycle",
        "• DAG Traversal",
        "• Data Flow (TensorDict)",
        "• Shape Management",
        "• Forward Execution",
        "• State Management",
    ]

    y_start = 4.2
    for i, resp in enumerate(responsibilities):
        ax.text(3, y_start - i * 0.3, resp, fontsize=9, ha="left", va="center")

    # ParamLayer extension
    param_box = FancyBboxPatch(
        (2, 0.5), 8, 1.2, boxstyle="round,pad=0.1", facecolor="lightblue", edgecolor="darkblue", linewidth=2
    )
    ax.add_patch(param_box)

    ax.text(6, 1.1, "ParamLayer extends LayerBase", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(6, 0.7, "+ Weight Management + Regularization", fontsize=10, ha="center", va="center")

    # Connection arrow
    arrow = ConnectionPatch(
        (6, 2), (6, 1.7), "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black"
    )
    ax.add_patch(arrow)

    # Testing challenges box
    test_box = FancyBboxPatch(
        (11, 2), 4, 4, boxstyle="round,pad=0.1", facecolor="lightyellow", edgecolor="orange", linewidth=2
    )
    ax.add_patch(test_box)

    ax.text(13, 5.5, "Testing Challenges", fontsize=12, fontweight="bold", ha="center", va="center", color="darkorange")

    challenges = [
        "• Full agent setup required",
        "• Complex TensorDict mocking",
        "• Integration tests only",
        "• 20+ methods to mock",
        "• Slow test execution",
    ]

    for i, challenge in enumerate(challenges):
        ax.text(11.2, 4.8 - i * 0.4, challenge, fontsize=9, ha="left", va="center")

    # Connection to challenges
    arrow2 = ConnectionPatch(
        (10, 4), (11, 4), "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="orange"
    )
    ax.add_patch(arrow2)

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Current Metta Architecture: Monolithic Design", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("current_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_proposed_architecture_diagram():
    """Create a diagram showing the proposed modular architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # System coordinator at top
    system_box = FancyBboxPatch(
        (4, 8), 6, 1.2, boxstyle="round,pad=0.1", facecolor="lightgreen", edgecolor="darkgreen", linewidth=2
    )
    ax.add_patch(system_box)
    ax.text(7, 8.6, "MettaSystem (Coordinator)", fontsize=12, fontweight="bold", ha="center", va="center")

    # Core components layer
    components = [
        ("MettaGraph\n(Structure)", 2, 6, "lightblue", "darkblue"),
        ("GraphExecutor\n(Execution)", 6, 6, "lightcyan", "darkcyan"),
        ("ShapePropagator\n(Validation)", 10, 6, "lightpink", "darkmagenta"),
    ]

    for name, x, y, facecolor, edgecolor in components:
        box = FancyBboxPatch(
            (x, y), 2.5, 1.5, boxstyle="round,pad=0.1", facecolor=facecolor, edgecolor=edgecolor, linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.75, name, fontsize=10, fontweight="bold", ha="center", va="center")

    # MettaModule at bottom
    module_box = FancyBboxPatch(
        (5, 3), 4, 2, boxstyle="round,pad=0.1", facecolor="lightsteelblue", edgecolor="navy", linewidth=2
    )
    ax.add_patch(module_box)
    ax.text(7, 4.5, "MettaModule", fontsize=14, fontweight="bold", ha="center", va="center")
    ax.text(7, 4, "(Pure Computation)", fontsize=10, ha="center", va="center")
    ax.text(7, 3.5, "tensor → tensor", fontsize=10, style="italic", ha="center", va="center")

    # Connections from system to components
    for _, x, y, _, _ in components:
        arrow = ConnectionPatch(
            (7, 8),
            (x + 1.25, y + 1.5),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc="gray",
            alpha=0.7,
        )
        ax.add_patch(arrow)

    # Connections from components to module
    for _, x, y, _, _ in components:
        arrow = ConnectionPatch(
            (x + 1.25, y),
            (7, 5),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc="gray",
            alpha=0.7,
        )
        ax.add_patch(arrow)

    # Benefits box
    benefits_box = FancyBboxPatch(
        (0.5, 0.5), 4, 4, boxstyle="round,pad=0.1", facecolor="lightgreen", edgecolor="darkgreen", linewidth=2
    )
    ax.add_patch(benefits_box)

    ax.text(2.5, 4, "Testing Benefits", fontsize=12, fontweight="bold", ha="center", va="center", color="darkgreen")

    benefits = [
        "✓ Unit testing possible",
        "✓ Fast test execution",
        "✓ Easy mocking",
        "✓ Isolated failures",
        "✓ Simple interfaces",
    ]

    for i, benefit in enumerate(benefits):
        ax.text(0.7, 3.4 - i * 0.4, benefit, fontsize=9, ha="left", va="center", color="darkgreen")

    # Architecture benefits box
    arch_benefits_box = FancyBboxPatch(
        (9.5, 0.5), 4, 4, boxstyle="round,pad=0.1", facecolor="lightcyan", edgecolor="darkcyan", linewidth=2
    )
    ax.add_patch(arch_benefits_box)

    ax.text(
        11.5, 4, "Architecture Benefits", fontsize=12, fontweight="bold", ha="center", va="center", color="darkcyan"
    )

    arch_benefits = [
        "✓ Single responsibility",
        "✓ Loose coupling",
        "✓ Easy extensions",
        "✓ Clear ownership",
        "✓ Maintainable code",
    ]

    for i, benefit in enumerate(arch_benefits):
        ax.text(9.7, 3.4 - i * 0.4, benefit, fontsize=9, ha="left", va="center", color="darkcyan")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Proposed Metta Architecture: Modular Design", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("proposed_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_testing_comparison():
    """Create a side-by-side comparison of testing approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Current testing approach
    ax1.text(
        0.5,
        0.95,
        "Current Testing (Integration Only)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax1.transAxes,
    )

    # Test setup complexity visualization
    setup_box = FancyBboxPatch(
        (0.1, 0.7),
        0.8,
        0.2,
        boxstyle="round,pad=0.02",
        facecolor="lightcoral",
        edgecolor="darkred",
        linewidth=1,
        transform=ax1.transAxes,
    )
    ax1.add_patch(setup_box)
    ax1.text(
        0.5,
        0.8,
        "Complex Test Setup\n50+ lines of configuration",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )

    # TensorDict complexity
    td_box = FancyBboxPatch(
        (0.1, 0.45),
        0.8,
        0.2,
        boxstyle="round,pad=0.02",
        facecolor="lightblue",
        edgecolor="darkblue",
        linewidth=1,
        transform=ax1.transAxes,
    )
    ax1.add_patch(td_box)
    ax1.text(
        0.5,
        0.55,
        "Manual TensorDict\nConstruction & Mocking",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )

    # Full integration
    integration_box = FancyBboxPatch(
        (0.1, 0.2),
        0.8,
        0.2,
        boxstyle="round,pad=0.02",
        facecolor="lightyellow",
        edgecolor="orange",
        linewidth=1,
        transform=ax1.transAxes,
    )
    ax1.add_patch(integration_box)
    ax1.text(
        0.5,
        0.3,
        "Full Integration Test\nSlow & Hard to Debug",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )

    # Proposed testing approach
    ax2.text(
        0.5,
        0.95,
        "Proposed Testing (Unit + Integration)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax2.transAxes,
    )

    # Unit tests
    unit_box = FancyBboxPatch(
        (0.1, 0.75),
        0.8,
        0.15,
        boxstyle="round,pad=0.02",
        facecolor="lightgreen",
        edgecolor="darkgreen",
        linewidth=1,
        transform=ax2.transAxes,
    )
    ax2.add_patch(unit_box)
    ax2.text(
        0.5, 0.825, "Fast Unit Tests\ntensor → tensor", fontsize=10, ha="center", va="center", transform=ax2.transAxes
    )

    # Component tests
    component_box = FancyBboxPatch(
        (0.1, 0.55),
        0.8,
        0.15,
        boxstyle="round,pad=0.02",
        facecolor="lightcyan",
        edgecolor="darkcyan",
        linewidth=1,
        transform=ax2.transAxes,
    )
    ax2.add_patch(component_box)
    ax2.text(
        0.5, 0.625, "Component Tests\nEasy Mocking", fontsize=10, ha="center", va="center", transform=ax2.transAxes
    )

    # Integration tests
    integration2_box = FancyBboxPatch(
        (0.1, 0.35),
        0.8,
        0.15,
        boxstyle="round,pad=0.02",
        facecolor="lightpink",
        edgecolor="darkmagenta",
        linewidth=1,
        transform=ax2.transAxes,
    )
    ax2.add_patch(integration2_box)
    ax2.text(
        0.5,
        0.425,
        "Targeted Integration\nFast & Focused",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )

    # Performance comparison
    perf_box = FancyBboxPatch(
        (0.1, 0.1),
        0.8,
        0.2,
        boxstyle="round,pad=0.02",
        facecolor="gold",
        edgecolor="darkorange",
        linewidth=1,
        transform=ax2.transAxes,
    )
    ax2.add_patch(perf_box)
    ax2.text(
        0.5,
        0.2,
        "Performance Improvement\n10x faster test execution\n95%+ test coverage",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    plt.suptitle("Testing Approach Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("testing_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_implementation_timeline():
    """Create a Gantt-style chart showing the implementation timeline."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    phases = [
        ("Foundation & Testing", 1, 2, "lightcoral"),
        ("Core Components", 3, 4, "lightblue"),
        ("Execution Layer", 5, 6, "lightgreen"),
        ("System Integration", 7, 8, "lightyellow"),
        ("Migration & Cleanup", 9, 10, "lightpink"),
    ]

    for i, (name, start, end, color) in enumerate(phases):
        ax.barh(i, end - start + 1, left=start - 1, height=0.6, color=color, edgecolor="black", linewidth=1)
        ax.text(start + (end - start) / 2 - 0.5, i, name, ha="center", va="center", fontweight="bold")

    # Add milestones
    milestones = [
        (2, "Testing Infrastructure Complete"),
        (4, "Core Architecture Ready"),
        (6, "Execution System Complete"),
        (8, "System Integration Complete"),
        (10, "Migration Complete"),
    ]

    for week, milestone in milestones:
        ax.plot([week - 0.5, week - 0.5], [-0.5, len(phases) - 0.5], "r--", linewidth=2, alpha=0.7)
        ax.text(week - 0.5, len(phases), milestone, rotation=45, ha="left", va="bottom", fontsize=9)

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([f"Phase {i + 1}" for i in range(len(phases))])
    ax.set_xlabel("Week")
    ax.set_title("Incremental Implementation Timeline", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 11)

    plt.tight_layout()
    plt.savefig("implementation_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all architecture diagrams."""
    print("Generating architecture diagrams...")

    try:
        create_current_architecture_diagram()
        print("✓ Current architecture diagram created: current_architecture.png")

        create_proposed_architecture_diagram()
        print("✓ Proposed architecture diagram created: proposed_architecture.png")

        create_testing_comparison()
        print("✓ Testing comparison diagram created: testing_comparison.png")

        create_implementation_timeline()
        print("✓ Implementation timeline created: implementation_timeline.png")

        print("\nAll diagrams generated successfully!")
        print("These can be included in your presentation to supervisors.")

    except Exception as e:
        print(f"Error generating diagrams: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    main()
