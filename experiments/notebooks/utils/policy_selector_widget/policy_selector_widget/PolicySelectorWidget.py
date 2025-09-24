"""
Policy Selector Widget for Jupyter Notebooks

An anywidget-based implementation for discovering and selecting policies.
Provides filtering by name/tags and supports multiselect functionality.
"""

import pathlib
from typing import Any, Callable, Dict, List

import anywidget
import traitlets

_DEV = False

bundled_assets_dir = pathlib.Path(__file__).parent / "static"
src_dir = pathlib.Path(__file__).parent / "../src"

if _DEV:
    ESM = "http://localhost:5176/src/index.js?anywidget"
else:
    ESM = (bundled_assets_dir / "index.js").read_text()

CSS = (src_dir / "styles.css").read_text()


class PolicySelectorWidget(anywidget.AnyWidget):
    """
    Interactive policy selector widget for discovering and selecting policies.

    Displays policies in a searchable list with multiselect functionality.
    Supports filtering by name, tags, and other policy metadata.
    """

    # AnyWidget requires _esm property for JavaScript code
    _esm = ESM
    _css = CSS
    name = traitlets.Unicode("PolicySelectorWidget").tag(sync=True)

    # Widget traits (data that syncs between Python and JavaScript)
    policy_data = traitlets.List([]).tag(sync=True)  # List of policy metadata dicts
    selected_policies = traitlets.List([]).tag(sync=True)  # List of selected policy IDs

    # Filter settings
    search_term = traitlets.Unicode("").tag(sync=True)
    policy_type_filter = traitlets.List([]).tag(sync=True)  # Filter by policy type
    tag_filter = traitlets.List([]).tag(sync=True)  # Filter by tags

    # UI configuration
    show_tags = traitlets.Bool(True).tag(sync=True)
    show_type = traitlets.Bool(True).tag(sync=True)
    show_created_at = traitlets.Bool(True).tag(sync=True)
    max_displayed_policies = traitlets.Int(100).tag(sync=True)

    # Event communication
    selection_changed = traitlets.Dict(allow_none=True, default_value=None).tag(
        sync=True
    )
    filter_changed = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    api_search_requested = traitlets.Dict(allow_none=True, default_value=None).tag(
        sync=True
    )
    api_search_completed = traitlets.Dict(allow_none=True, default_value=None).tag(
        sync=True
    )
    load_all_policies_requested = traitlets.Bool(False).tag(sync=True)

    # Alternative: use a simple counter that changes on each search request
    search_trigger = traitlets.Int(0).tag(sync=True)
    current_search_params = traitlets.Dict(allow_none=True, default_value=None).tag(
        sync=True
    )

    # Search configuration
    use_api_search = traitlets.Bool(True).tag(sync=True)  # Enable API-based search
    search_debounce_ms = traitlets.Int(300).tag(
        sync=True
    )  # Debounce delay in milliseconds

    def __init__(self, client=None, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {
            "selection_changed": [],
            "filter_changed": [],
            "api_search_requested": [],
        }
        self._client = None

        # Set client if provided
        if client is not None:
            self.set_client(client)

        # Set up observers
        self.observe(self._on_selection_changed, names="selection_changed")
        self.observe(self._on_filter_changed, names="filter_changed")
        self.observe(self._on_api_search_requested, names="api_search_requested")
        self.observe(self._on_search_trigger, names="search_trigger")
        self.observe(
            self._on_load_all_policies_requested, names="load_all_policies_requested"
        )

    def _on_selection_changed(self, change):
        """Handle policy selection events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["selection_changed"]:
                callback(change["new"])

    def _on_filter_changed(self, change):
        """Handle filter change events from JavaScript."""
        if change["new"]:
            for callback in self._callbacks["filter_changed"]:
                callback(change["new"])

    def _on_api_search_requested(self, change):
        """Handle API search requests from JavaScript."""
        if change["new"]:
            search_params = change["new"]
            # Check if client is configured
            if not self._client:
                print("⚠️ No client configured - cannot perform API search")
                return

            # Try synchronous search first - simpler and more reliable in Jupyter
            try:
                print("🔄 Attempting synchronous search...")

                search_term = search_params.get("searchTerm", "")
                policy_type = (
                    search_params.get("policyTypeFilter", [None])[0]
                    if search_params.get("policyTypeFilter")
                    else None
                )
                tags = search_params.get("tagFilter")

                print(
                    f"🔍 Searching for: term='{search_term}', type={policy_type}, tags={tags}"
                )

                # Use sync version of client method if available, otherwise fall back to async in thread
                if hasattr(self._client, "search_policies_sync"):
                    print("🔄 Using synchronous client method...")
                    response = self._client.search_policies_sync(
                        search=search_term or None,
                        policy_type=policy_type,
                        tags=tags,
                    )

                    # Convert response to widget format
                    policies = [
                        {
                            "id": p.id,
                            "type": p.type,
                            "name": p.name,
                            "user_id": p.user_id,
                            "created_at": p.created_at,
                            "tags": p.tags,
                        }
                        for p in response.policies
                    ]

                    self.policy_data = list(policies)

                    # Signal search completion to React
                    completion_signal = {
                        "searchTerm": search_term,
                        "resultCount": len(policies),
                        "timestamp": __import__("time").time(),
                    }
                    self.api_search_completed = completion_signal

                else:
                    print("🔄 No sync method, falling back to async in thread...")
                    # Fall back to async in thread
                    import asyncio
                    import threading

                    def run_async_search():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                self._handle_api_search(search_params)
                            )
                            print("🟢 Async search thread completed")
                            return result
                        except Exception as e:
                            print(f"🚨 Error in async search thread: {e}")
                            import traceback

                            print(traceback.format_exc())
                            # Force update to stop spinner
                            current_data = (
                                list(self.policy_data) if self.policy_data else []
                            )
                            self.policy_data = current_data
                        finally:
                            loop.close()

                    search_thread = threading.Thread(
                        target=run_async_search, daemon=True
                    )
                    search_thread.start()

            except Exception as e:
                print(f"🚨 Error in search handler: {e}")
                import traceback

                print(traceback.format_exc())
                # Force update to stop spinner
                current_data = list(self.policy_data) if self.policy_data else []
                self.policy_data = current_data

            for callback in self._callbacks["api_search_requested"]:
                callback(change["new"])

    def _on_search_trigger(self, change):
        """Handle search trigger events from JavaScript (alternative approach)."""
        print(f"🔔 _on_search_trigger called with change: {change}")
        if change["new"] > 0 and self.current_search_params:
            search_params = self.current_search_params
            print(f"🔍 Search triggered via counter: {search_params}")

            # Check if client is configured
            if not self._client:
                print("⚠️ No client configured - cannot perform API search")
                return

            # Use the same search logic as the main handler
            try:
                print("🔄 Attempting synchronous search (via trigger)...")

                search_term = search_params.get("searchTerm", "")
                policy_type = (
                    search_params.get("policyTypeFilter", [None])[0]
                    if search_params.get("policyTypeFilter")
                    else None
                )
                tags = search_params.get("tagFilter")

                print(
                    f"🔍 Searching for: term='{search_term}', type={policy_type}, tags={tags}"
                )

                if hasattr(self._client, "search_policies_sync"):
                    print("🔄 Using synchronous client method...")
                    response = self._client.search_policies_sync(
                        search=search_term or None,
                        policy_type=policy_type,
                        tags=tags,
                    )

                    # Convert response to widget format
                    policies = [
                        {
                            "id": p.id,
                            "type": p.type,
                            "name": p.name,
                            "user_id": p.user_id,
                            "created_at": p.created_at,
                            "tags": p.tags,
                        }
                        for p in response.policies
                    ]

                    print(f"📊 Sync search returned {len(policies)} results")
                    self.policy_data = list(policies)
                    # Signal search completion to React
                    completion_signal = {
                        "searchTerm": search_term,
                        "resultCount": len(policies),
                        "timestamp": __import__("time").time(),
                    }
                    print(f"📤 Sending completion signal to React: {completion_signal}")
                    self.api_search_completed = completion_signal
                    print("✅ Sync search completed successfully")

            except Exception as e:
                print(f"🚨 Error in search trigger handler: {e}")
                import traceback

                print(traceback.format_exc())

    def _on_load_all_policies_requested(self, change):
        """Handle request to load all policies from the client."""
        if change["new"]:
            # Check if client is configured
            if not self._client:
                return

            try:
                # Use search with no filters to get all policies
                response = self._client.search_policies_sync(
                    search=None,
                    policy_type=None,
                    tags=None,
                    limit=1000,  # Get more policies by default
                    offset=0,
                )

                # Convert response to widget format
                policies = [
                    {
                        "id": p.id,
                        "type": p.type,
                        "name": p.name,
                        "user_id": p.user_id,
                        "created_at": p.created_at,
                        "tags": p.tags,
                    }
                    for p in response.policies
                ]

                self.policy_data = list(policies)

            except Exception:
                import traceback

                print(traceback.format_exc())
            finally:
                # Reset the trigger
                self.load_all_policies_requested = False

    def on_selection_changed(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for when policy selection changes.

        Args:
            callback: Function that receives {'selected_policies': List[str], 'action': str}
        """
        self._callbacks["selection_changed"].append(callback)

    def on_filter_changed(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for when filters change.

        Args:
            callback: Function that receives filter change information
        """
        self._callbacks["filter_changed"].append(callback)

    def on_api_search_requested(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for when API search is requested.

        Args:
            callback: Function that receives search parameters
        """
        self._callbacks["api_search_requested"].append(callback)

    def set_policy_data(self, policies: List[Dict[str, Any]]):
        """Set the policy selector data.

        Args:
            policies: List of policy metadata dicts with keys like:
                     - id: Policy ID
                     - name: Policy name
                     - type: Policy type ('training_run' or 'policy')
                     - tags: List of tags
                     - created_at: Creation timestamp
                     - user_id: Optional user ID
        """
        self.policy_data = policies

    def set_search_term(self, term: str):
        """Set the search filter term.

        Args:
            term: Search string to filter policies by name
        """
        self.search_term = term

    def set_policy_type_filter(self, types: List[str]):
        """Set the policy type filter.

        Args:
            types: List of policy types to include (empty means all)
        """
        self.policy_type_filter = types

    def set_tag_filter(self, tags: List[str]):
        """Set the tag filter.

        Args:
            tags: List of tags to include (empty means all)
        """
        self.tag_filter = tags

    def get_selected_policies(self) -> List[str]:
        """Get the currently selected policy IDs.

        Returns:
            List of selected policy IDs
        """
        return self.selected_policies

    def select_policies(self, policy_ids: List[str]):
        """Programmatically select policies.

        Args:
            policy_ids: List of policy IDs to select
        """
        self.selected_policies = policy_ids

    def clear_selection(self):
        """Clear all selected policies."""
        self.selected_policies = []

    def select_all_filtered(self):
        """Select all policies that match current filters."""
        # This will be handled by the JavaScript side
        pass

    def set_ui_config(
        self,
        show_tags: bool = True,
        show_type: bool = True,
        show_created_at: bool = True,
        max_displayed_policies: int = 100,
    ):
        """Configure UI display options.

        Args:
            show_tags: Whether to show policy tags
            show_type: Whether to show policy type
            show_created_at: Whether to show creation date
            max_displayed_policies: Maximum number of policies to display
        """
        self.show_tags = show_tags
        self.show_type = show_type
        self.show_created_at = show_created_at
        self.max_displayed_policies = max_displayed_policies

    def get_policy_by_id(self, policy_id: str) -> Dict[str, Any] | None:
        """Get policy metadata by ID.

        Args:
            policy_id: Policy ID to look up

        Returns:
            Policy metadata dict or None if not found
        """
        for policy in self.policy_data:
            if policy.get("id") == policy_id:
                return policy
        return None

    def get_selected_policy_data(self) -> List[Dict[str, Any]]:
        """Get full metadata for currently selected policies.

        Returns:
            List of policy metadata dicts for selected policies
        """
        return [
            policy
            for policy in self.policy_data
            if policy.get("id") in self.selected_policies
        ]

    def set_client(self, client):
        """Set the client for API-based operations.

        Args:
            client: ScorecardClient instance for making API requests, or None to remove client
        """
        # Validate client has required methods if not None
        if client is not None:
            if not hasattr(client, "search_policies"):
                raise ValueError(
                    "Client must have a 'search_policies' method (expected ScorecardClient)"
                )

        self._client = client
        # Automatically enable API search when client is set
        if client is not None:
            self.use_api_search = True
        else:
            self.use_api_search = False

    def get_client(self):
        """Get the currently configured client.

        Returns:
            The configured ScorecardClient or None if not set
        """
        return self._client

    async def load_policies_from_client(self, limit: int = 100):
        """Load policies from the configured client.

        Args:
            limit: Maximum number of policies to load

        Returns:
            List of loaded policies
        """
        if not self._client:
            print(
                "⚠️ No client configured. Use widget.set_client(scorecard_client) first."
            )
            return []

        try:
            policies = await self.search_policies_async(limit=limit)
            self.set_policy_data(policies)
            return policies
        except Exception:
            return []

    async def search_policies_async(
        self,
        search: str | None = None,
        policy_type: str | None = None,
        tags: List[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Perform async search using the configured client.

        Args:
            search: Search term for policy names
            policy_type: Filter by policy type
            tags: Filter by tags
            limit: Maximum results
            offset: Results offset

        Returns:
            List of policy dictionaries
        """
        if not self._client:
            print(
                "⚠️ No search client configured. Use widget.set_client(scorecard_client) to enable API search."
            )
            print("   Falling back to local data filtering")
            return self.policy_data

        try:
            response = await self._client.search_policies(
                search=search,
                policy_type=policy_type,
                tags=tags,
                limit=limit,
                offset=offset,
            )

            # Convert response to widget format
            policies = [
                {
                    "id": p.id,
                    "type": p.type,
                    "name": p.name,
                    "user_id": p.user_id,
                    "created_at": p.created_at,
                    "tags": p.tags,
                }
                for p in response.policies
            ]

            return policies

        except Exception as e:
            print(f"🚨 Search API error: {e}, falling back to local data")
            return self.policy_data

    async def _handle_api_search(self, search_params: Dict[str, Any]):
        """Handle API search request automatically."""
        search_term = search_params.get("searchTerm", "")
        policy_type = (
            search_params.get("policyTypeFilter", [None])[0]
            if search_params.get("policyTypeFilter")
            else None
        )
        tags = search_params.get("tagFilter")

        if not self._client:
            print("❌ No search client configured!")
            return

        try:
            results = await self.search_policies_async(
                search=search_term,  # Convert empty string to None
                policy_type=policy_type,
                tags=tags,
            )

            # Force update the policy data to trigger React re-render
            # Use a new list to ensure trait change is detected
            self.policy_data = list(results)

        except Exception as e:
            print(f"🚨 API search failed: {e}")
            import traceback

            print(traceback.format_exc())

            # Even on failure, force update the policy data to ensure spinner stops
            # Create a new list reference to trigger trait change
            current_data = list(self.policy_data) if self.policy_data else []
            self.policy_data = current_data
            print("🔄 Forced policy data refresh to stop spinner")


def create_policy_selector_widget(client=None, **kwargs) -> PolicySelectorWidget:
    """Create and return a new policy selector widget.

    Args:
        client: Optional ScorecardClient instance for API-based operations
        **kwargs: Additional keyword arguments passed to PolicySelectorWidget constructor

    Returns:
        PolicySelectorWidget instance
    """
    return PolicySelectorWidget(client=client, **kwargs)
