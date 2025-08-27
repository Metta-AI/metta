import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class NotebookWidgetsSetup(SetupModule):
    install_once = False

    _widgets = [
        "scorecard_widget",
        "eval_finder_widget",
        "policy_selector_widget",
    ]

    _widget_root = Path("experiments/notebooks/utils")

    def dependencies(self) -> list[str]:
        return ["nodejs"]

    @property
    def description(self) -> str:
        return "The python notebook widgets we create"

    def __init__(self):
        super().__init__()
        self.widget_root = self.repo_root / self._widget_root

    def should_install_widget(self, widget: str) -> bool:
        widget_path = self.widget_root / widget
        node_modules_path = widget_path / "node_modules"
        return not node_modules_path.exists()

    def should_build_widget(self, widget: str) -> bool:
        widget_path = self.widget_root / widget
        build_cache_status = subprocess.run(
            [
                "bash",
                "-c",
                "pnpm exec turbo run build --dry=json 2>/dev/null | jq .tasks[0].cache.status",
            ],
            cwd=widget_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        build_cache_status = build_cache_status.stdout.decode("utf-8").strip()
        # INFO: https://turborepo.com/docs/reference/run#--dry----dry-run
        # This will be "MISS" if the project needs to be built or "HIT" if not.
        return build_cache_status == '"MISS"'

    def check_installed(self) -> bool:
        for widget in self._widgets:
            if self.should_install_widget(widget):
                return False
            if self.should_build_widget(widget):
                return False
        return True

    def install(self) -> None:
        info("Setting up Metta's custom Python notebook widgets...")
        try:
            installed_count = 0
            built_count = 0

            for widget in self._widgets:
                widget_path = self.widget_root / widget

                if self.should_install_widget(widget):
                    info(f"Installing dependencies for {widget}...")
                    subprocess.run(
                        ["pnpm", "install"],
                        check=True,
                        cwd=widget_path,
                    )
                    installed_count += 1

                    # After installing, always build
                    info(f"Building {widget}...")
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            "pnpm install && pnpm exec turbo run build",
                        ],
                        check=True,
                        cwd=widget_path,
                    )
                    built_count += 1

                elif self.should_build_widget(widget):
                    info(f"Building {widget} (cache miss)...")
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            "pnpm exec turbo run build",
                        ],
                        check=True,
                        cwd=widget_path,
                    )
                    built_count += 1
                else:
                    info(f"Skipping {widget} (up to date)")

            if installed_count > 0:
                success(f"Installed dependencies for {installed_count} widget(s)")
            if built_count > 0:
                success(f"Built {built_count} widget(s)")

            success("Notebook widgets are ready!")
            info(
                "Check out ./experiments/notebooks/*_example.ipynb to see them in action.\n"
                "Import in your notebooks like:\n"
                "`from experiments.notebooks.utils.scorecard_widget.scorecard_widget."
                "ScorecardWidget import ScorecardWidget`"
            )

        except subprocess.CalledProcessError as e:
            warning(f"NotebookWidgets compilation failed: {e}")
            info("You can compile them manually:")
            for widget in self._widgets:
                info(f"1. cd {self._widget_root / widget}")
                info("2. pnpm install && npx turbo run build")
                info("---")
