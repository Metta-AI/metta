"""Interaction tests for miniscope renderer components."""

from __future__ import annotations

from dataclasses import dataclass

from mettagrid.renderer.miniscope.buffer import MapBuffer
from mettagrid.renderer.miniscope.components.glyph_picker import GlyphPickerComponent
from mettagrid.renderer.miniscope.miniscope import MiniscopeRenderer
from mettagrid.renderer.miniscope.miniscope_state import (
    RenderMode,
)


def _minimal_renderer() -> MiniscopeRenderer:
    """Create a renderer with the minimal wiring for input handling tests."""

    renderer = MiniscopeRenderer()
    renderer._env = object()  # Input handling checks only require a non-None sentinel
    renderer._components = []
    return renderer


def test_sidebar_toggle_via_numeric_keys() -> None:
    """Numeric hotkeys should toggle sidebar panel visibility."""

    renderer = _minimal_renderer()
    renderer._sidebar_hotkeys = {"1": "agent_info"}
    renderer._state.sidebar_visibility = {"agent_info": True}
    renderer._state.mode = RenderMode.FOLLOW

    renderer._state.user_input = "1"
    renderer._handle_user_input()
    assert renderer._state.sidebar_visibility["agent_info"] is False

    renderer._state.user_input = "1"
    renderer._handle_user_input()
    assert renderer._state.sidebar_visibility["agent_info"] is True


def test_help_modal_entry_and_exit() -> None:
    """The help modal should open on '?' and close on any subsequent key."""

    renderer = _minimal_renderer()
    renderer._state.initialize_sidebar_visibility(["agent_info", "object_info", "symbols", "glyph_picker", "help"])
    renderer._sidebar_hotkeys = {}

    renderer._state.mode = RenderMode.FOLLOW
    renderer._state.user_input = "?"
    renderer._handle_user_input()

    assert renderer._state.mode == RenderMode.HELP
    assert renderer._state.is_sidebar_visible("help") is True
    assert renderer._state.is_sidebar_visible("agent_info") is False

    renderer._state.user_input = "x"
    renderer._handle_user_input()

    assert renderer._state.mode == RenderMode.FOLLOW
    assert renderer._state.is_sidebar_visible("agent_info") is True
    assert renderer._state.is_sidebar_visible("help") is False


@dataclass
class _DummyEnv:
    action_names: list[str]
    num_agents: int


class _DummyComponent:
    def __init__(self) -> None:
        self.handled: list[str] = []

    def handle_input(self, ch: str) -> bool:  # noqa: D401 - simple stub
        self.handled.append(ch)
        return False


def test_glyph_picker_captures_input_exclusively() -> None:
    """Glyph picker mode should intercept input before other components."""

    renderer = _minimal_renderer()
    renderer._state.initialize_sidebar_visibility(["glyph_picker"])
    renderer._panels.register_sidebar_panel("glyph_picker")

    env = _DummyEnv(action_names=["noop", "change_glyph_0"], num_agents=1)
    renderer._env = env

    glyph_picker = GlyphPickerComponent(env=env, state=renderer._state, panels=renderer._panels)
    other_component = _DummyComponent()
    renderer._components = [glyph_picker, other_component]

    renderer._state.mode = RenderMode.GLYPH_PICKER
    renderer._state.user_input = "a"
    renderer._handle_user_input()

    assert glyph_picker._glyph_query == "a"
    assert other_component.handled == []


def test_highlighted_agent_renders_star() -> None:
    """Highlighted agents should be rendered with a star symbol."""

    object_type_names = ["agent"]
    symbol_map = {"agent": "A", "empty": "."}
    buffer = MapBuffer(object_type_names, symbol_map)

    grid_objects = {1: {"r": 0, "c": 0, "type_name": "agent", "agent_id": 7}}
    buffer.set_highlighted_agent(7)

    rendered = buffer.render(grid_objects, use_viewport=False)
    assert "⭐" in rendered


def test_select_mode_shows_object_info() -> None:
    """Entering SELECT mode should automatically show object_info panel."""

    renderer = _minimal_renderer()
    renderer._state.initialize_sidebar_visibility(["agent_info", "object_info", "symbols"])

    # Hide object_info initially
    renderer._state.sidebar_visibility["object_info"] = False
    assert renderer._state.is_sidebar_visible("object_info") is False

    # Enter SELECT mode
    renderer._state.set_mode(RenderMode.SELECT)

    # object_info should now be visible
    assert renderer._state.is_sidebar_visible("object_info") is True
    assert renderer._state.mode == RenderMode.SELECT


def test_map_expands_when_sidebar_hidden() -> None:
    """Map should use full width when all sidebar panels are hidden."""

    renderer = _minimal_renderer()
    renderer._initial_terminal_columns = 120
    renderer._initial_terminal_lines = 40
    renderer._state.initialize_sidebar_visibility(["agent_info", "object_info", "symbols"])
    renderer._state.map_height = 30
    renderer._state.map_width = 50

    from rich.console import Console

    from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout

    renderer._panels = PanelLayout(Console())

    # Initially some panels are visible
    renderer._state.sidebar_visibility["agent_info"] = True
    renderer._update_viewport_size()
    map_width_with_sidebar = renderer._panels.map_view.width

    # Hide all sidebar panels
    renderer._state.sidebar_visibility["agent_info"] = False
    renderer._state.sidebar_visibility["object_info"] = False
    renderer._state.sidebar_visibility["symbols"] = False
    renderer._update_viewport_size()
    map_width_without_sidebar = renderer._panels.map_view.width

    # Map should be wider when sidebar is hidden
    assert map_width_without_sidebar > map_width_with_sidebar


def test_help_modal_restores_sidebar_state() -> None:
    """Help modal should restore previous sidebar visibility state on exit."""

    renderer = _minimal_renderer()
    renderer._state.initialize_sidebar_visibility(["agent_info", "object_info", "symbols", "help"])

    # Set specific visibility state
    renderer._state.sidebar_visibility["agent_info"] = True
    renderer._state.sidebar_visibility["object_info"] = False
    renderer._state.sidebar_visibility["symbols"] = True
    renderer._state.sidebar_visibility["help"] = False

    # Save the state to compare later
    original_state = renderer._state.sidebar_visibility.copy()

    # Enter help mode
    renderer._state.enter_help()
    assert renderer._state.mode == RenderMode.HELP
    assert renderer._state.is_sidebar_visible("help") is True
    assert renderer._state.is_sidebar_visible("agent_info") is False

    # Exit help mode
    renderer._state.exit_help()

    # Sidebar should be restored to original state
    assert renderer._state.sidebar_visibility["agent_info"] == original_state["agent_info"]
    assert renderer._state.sidebar_visibility["object_info"] == original_state["object_info"]
    assert renderer._state.sidebar_visibility["symbols"] == original_state["symbols"]
    assert renderer._state.sidebar_visibility["help"] == original_state["help"]


def test_glyph_picker_restores_sidebar_state() -> None:
    """Glyph picker modal should restore previous sidebar visibility state on exit."""

    renderer = _minimal_renderer()
    renderer._state.initialize_sidebar_visibility(["agent_info", "object_info", "symbols", "glyph_picker"])

    # Set specific visibility state
    renderer._state.sidebar_visibility["agent_info"] = False
    renderer._state.sidebar_visibility["object_info"] = True
    renderer._state.sidebar_visibility["symbols"] = False
    renderer._state.sidebar_visibility["glyph_picker"] = False

    # Save the state to compare later
    original_state = renderer._state.sidebar_visibility.copy()

    # Enter glyph picker mode
    renderer._state.enter_glyph_picker()
    assert renderer._state.mode == RenderMode.GLYPH_PICKER
    assert renderer._state.is_sidebar_visible("glyph_picker") is True
    assert renderer._state.is_sidebar_visible("agent_info") is True
    assert renderer._state.is_sidebar_visible("object_info") is False

    # Exit glyph picker mode
    renderer._state.exit_glyph_picker()

    # Sidebar should be restored to original state
    assert renderer._state.sidebar_visibility["agent_info"] == original_state["agent_info"]
    assert renderer._state.sidebar_visibility["object_info"] == original_state["object_info"]
    assert renderer._state.sidebar_visibility["symbols"] == original_state["symbols"]
    assert renderer._state.sidebar_visibility["glyph_picker"] == original_state["glyph_picker"]
