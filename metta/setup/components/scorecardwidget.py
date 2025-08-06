import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class ScorecardWidgetSetup(SetupModule):
    install_once = False

    def dependencies(self) -> list[str]:
        return ["nodejs"]

    @property
    def description(self) -> str:
        return "The policy <Scorecard /> component from Observatory, implemented as a Jupyter notebook widget"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("scorecardwidget")

    def check_installed(self) -> bool:
        has_node_modules = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/scorecard_widget/node_modules"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        has_compiled_js = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/scorecard_widget/scorecard_widget/static/index.js"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        should_build = (
            subprocess.call(
                ["bash", "-c", "./should_build.sh"],
                cwd=self.repo_root / "experiments/notebooks/utils/scorecard_widget",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        return has_node_modules and has_compiled_js and not should_build

    def install(self) -> None:
        info("Setting up ScorecardWidget...")
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

            info(
                "The ScorecardWidget is now compiled. Check out "
                "./experiments/notebooks/scorecard_widget_example.ipynb "
                "to see it in action and learn how to use it. "
                "\n"
                "You can also use it in your own notebooks by importing it like "
                "\n"
                "`from experiments.notebooks.utils.scorecard_widget.scorecard_widget.ScorecardWidget import "
                "ScorecardWidget`."
            )

        except subprocess.CalledProcessError:
            warning("""
                ScorecardWidget compilation failed. You can compile it manually.
                1. cd ./experiments/notebooks/utils/scorecard_widget
                2. npm install
                3. npm run build
            """)
