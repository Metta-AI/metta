import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class NotebookWidgetsSetup(SetupModule):
    install_once = False

    def dependencies(self) -> list[str]:
        return ["nodejs"]

    @property
    def description(self) -> str:
        return "The python notebook widgets we create"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("notebookwidgets")

    def check_installed(self) -> bool:
        scorecard_node_modules = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/scorecard_widget/node_modules"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        eval_finder_node_modules = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/eval_finder_widget/node_modules"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        scorecard_compiled_js = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/scorecard_widget/scorecard_widget/static/index.js"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        eval_finder_compiled_js = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/eval_finder_widget/eval_finder_widget/static/index.js"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        scorecard_should_build = (
            subprocess.call(
                ["bash", "-c", "./should_build.sh"],
                cwd=self.repo_root / "experiments/notebooks/utils/scorecard_widget/scorecard_widget",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        eval_finder_should_build = (
            subprocess.call(
                ["bash", "-c", "./should_build.sh"],
                cwd=self.repo_root / "experiments/notebooks/utils/scorecard_widget",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        has_node_modules = scorecard_node_modules and eval_finder_node_modules
        has_compiled_js = scorecard_compiled_js and eval_finder_compiled_js
        should_build = scorecard_should_build and eval_finder_should_build
        return has_node_modules and has_compiled_js and not should_build

    def install(self) -> None:
        info("Setting up Metta's custom Python notebook widgets...")
        try:
            if not self.check_installed():
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        "npm install && npm run build",
                    ],
                    check=True,
                    cwd=self.repo_root / "experiments/notebooks/utils/scorecard_widget",
                )
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        "npm install && npm run build",
                    ],
                    check=True,
                    cwd=self.repo_root / "experiments/notebooks/utils/eval_finder_widget",
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
                2. npm install
                3. npm run build
                4. cd ./experiments/notebooks/utils/eval_finder_widget
                5. npm install
                6. npm run build
            """)
