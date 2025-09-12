import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class NotebookWidgetsSetup(SetupModule):
    install_once = False

    _widgets = [
        "scorecard_widget",
        "eval_finder_widget",
        "policy_selector_widget",
    ]

    def dependencies(self) -> list[str]:
        return ["nodejs"]

    @property
    def description(self) -> str:
        return "The python notebook widgets we create"

    def __init__(self):
        super().__init__()
        self.widget_root = self.repo_root / "experiments/notebooks/utils"

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

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Setting up Metta's custom Python notebook widgets...")
        try:
            for widget in self._widgets:
                if self.should_install_widget(widget) or force:
                    print(f"Installing dependencies and building {widget}...")
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            "pnpm install && pnpm exec turbo run build",
                        ],
                        check=True,
                        cwd=self.widget_root / widget,
                    )
                    continue
                if self.should_build_widget(widget) or force:
                    print(f"Building {widget} (cache miss)...")
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            "pnpm exec turbo run build",
                        ],
                        check=True,
                        cwd=self.widget_root / widget,
                    )
            info(
                "The notebook widgets are now compiled. Check out "
                "./experiments/notebooks/*_example.ipynb "
                "to see them in action and learn how to use them. "
                "\n"
                "You can also use them in your own notebooks by importing them like "
                "\n"
                "`from experiments.notebooks.utils.scorecard_widget.scorecard_widget.ScorecardWidget import "
                "ScorecardWidget`."
            )

        except subprocess.CalledProcessError:
            warning("""
                NotebookWidgets compilation failed. You can compile them manually:
                1. cd ./experiments/notebooks/utils/scorecard_widget
                2. pnpm install
                3. pnpm run build
                4. cd ./experiments/notebooks/utils/eval_finder_widget
                5. pnpm install
                6. pnpm run build
            """)
