import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class HeatmapWidgetSetup(SetupModule):
    install_once = True
    # install_once = False

    @property
    def description(self) -> str:
        return "The policy <Heatmap /> component from Observatory, implemented as a Jupyter notebook widget"

    def is_applicable(self) -> bool:
        return subprocess.call(["which", "npm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    def check_installed(self) -> bool:
        has_compiled_js = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/heatmap_widget/heatmap_widget/static/index.js"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        has_heatmap_tsx = (
            subprocess.call(
                ["ls", "./experiments/notebooks/utils/heatmap_widget/src/Heatmap.tsx"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        return has_compiled_js and has_heatmap_tsx

    def install(self) -> None:
        info("Setting up HeatmapWidget...")
        try:
            subprocess.run(
                [
                    "bash",
                    "-c",
                    "cp observatory/src/Heatmap.tsx experiments/notebooks/utils/heatmap_widget/src/Heatmap.tsx",
                ],
                check=True,
                cwd=self.repo_root,
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    "npm install && npm run build",
                ],
                check=True,
                cwd=self.repo_root / "experiments/notebooks/utils/heatmap_widget",
            )

            info(
                "The HeatmapWidget is now compiled. Check out "
                "./experiments/notebooks/heatmap_widget_example.ipynb "
                "to see it in action and learn how to use it. "
                "\n"
                "You can also use it in your own notebooks by importing it like "
                "\n"
                "`from experiments.notebooks.utils.heatmap_widget.heatmap_widget.HeatmapWidget import "
                "HeatmapWidget`."
            )

        except subprocess.CalledProcessError:
            warning("""
                HeatmapWidget compilation failed. You can compile it manually.
                1. cd ./experiments/notebooks/utils/heatmap_widget
                2. npm install
                3. npm run build
            """)
