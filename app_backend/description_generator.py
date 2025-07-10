import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import httpx

from app_backend.git_client import GitClient, GitCommit, GitError
from app_backend.github_client import GitHubClient
from app_backend.llm_client import LLMClient

logger = logging.getLogger("description_generator")


class TrainingRunInput(NamedTuple):
    """Input data for training run description generation."""

    commits: List[GitCommit]
    diff_stats: str
    diff_content: str


class GitDataProvider:
    """Wrapper that provides git data with configurable fallback between local and GitHub clients."""

    def __init__(self, http_client: httpx.Client, mode: str = "auto"):
        """
        Initialize the git data provider.

        Args:
            http_client: HTTP client for GitHub operations
            mode: Operation mode - "auto", "local_only", "github_only"
        """
        if mode not in ("auto", "local_only", "github_only"):
            raise ValueError(f"Invalid git client mode: {mode}")

        # Assign clients based on mode
        if mode == "github_only":
            self.git_client = None
            self.github_client = GitHubClient(http_client)
        elif mode == "local_only":
            self.git_client = GitClient()
            self.github_client = None
        else:  # mode == "auto"
            self.git_client = GitClient()
            self.github_client = GitHubClient(http_client)

    def get_commit_data(self, commit_hash: str) -> Tuple[List[GitCommit], str, str]:
        """
        Get commit data (commits, diff_stats, diff_content) with fallback.

        Args:
            commit_hash: The git commit hash

        Returns:
            Tuple of (commits, diff_stats, diff_content)

        Raises:
            ValueError: If data cannot be retrieved from any configured source
        """
        # Try git client first if available
        if self.git_client is not None:
            try:
                commits = self.git_client.get_commit_range(commit_hash)
                diff_stats = self.git_client.get_range_diff_stats(commit_hash)
                diff_content = self.git_client.get_range_diff(commit_hash)
                return commits, diff_stats, diff_content
            except (GitError, ValueError):
                pass

        # Try GitHub client if available
        if self.github_client is not None:
            try:
                _, commits, diff_stats, diff_content = self.github_client.get_all_commit_data(commit_hash)
                return commits, diff_stats, diff_content
            except Exception:
                pass

        raise ValueError(f"Commit {commit_hash} not found with any configured git client")


class TrainingRunDescriptionGenerator:
    """
    Generator for training run descriptions using LLM APIs.
    Follows the existing client pattern with dependency injection.
    """

    def __init__(self, git_data_provider: GitDataProvider, llm_client: Optional[LLMClient] = None):
        """
        Initialize the training run description generator.

        Args:
            git_data_provider: Provider for git data with configurable fallback
            llm_client: Optional LLM client for generating descriptions
        """
        self.git_data_provider = git_data_provider
        self.llm_client = llm_client

        # Check if LLM client is available
        if llm_client is None or not llm_client.is_available():
            logger.warning("No LLM client available - will fall back to git logs only")

    def generate_description(self, commit_hash: str, has_uncommitted_changes: bool = False) -> str:
        """
        Generate a training run description based on git commit history.

        Args:
            commit_hash: The git commit hash for this training run
            has_uncommitted_changes: Whether the commit had uncommitted changes

        Returns:
            Generated description text

        Raises:
            Various exceptions propagated from git/github operations
        """
        commits, diff_stats, diff_content = self.git_data_provider.get_commit_data(commit_hash)

        if not commits:
            raise ValueError(f"No commits found for hash {commit_hash}")

        # Build input data
        input_data = TrainingRunInput(commits=commits, diff_stats=diff_stats, diff_content=diff_content)

        # Check if LLM is available
        if self.llm_client and self.llm_client.is_available():
            # Use LLM to generate description
            messages = self._build_llm_messages(input_data)
            logger.debug(f"Using LLM to generate description for commit {commit_hash[:8]}")
            logger.debug(f"LLM conversation with {len(messages)} messages")

            try:
                result = self.llm_client.generate_text_with_messages(messages)
                logger.debug(f"LLM generation successful, result length: {len(result)}")
                return self._add_uncommitted_changes_disclaimer(result, commit_hash, has_uncommitted_changes)
            except Exception as e:
                logger.warning(f"LLM generation failed, falling back to git logs: {e}")

        # Fallback to formatted git logs
        logger.debug(f"Using git logs fallback for commit {commit_hash[:8]} (LLM not available)")
        result = self._build_git_summary(commits)

        return self._add_uncommitted_changes_disclaimer(result, commit_hash, has_uncommitted_changes)

    def _format_user_message(self, input_data: TrainingRunInput) -> str:
        """
        Format a user message from training run input data.

        Args:
            input_data: TrainingRunInput containing commits, diff_stats, and diff_content

        Returns:
            Formatted user message string
        """
        # Build commit history section without hashes
        commit_details = []
        for commit in input_data.commits:
            commit_details.append(f"- {commit.message} (by {commit.author} on {commit.date})")

        commit_history = "\n".join(commit_details)

        # Limit actual diff to reasonable size (50KB max)
        MAX_DIFF_SIZE = 50000
        diff_content = input_data.diff_content
        if len(diff_content) > MAX_DIFF_SIZE:
            # Keep first part and add truncation notice
            diff_content = diff_content[:MAX_DIFF_SIZE] + "\n... (diff truncated for brevity)"
            logger.debug(f"Actual diff truncated to {MAX_DIFF_SIZE} chars")

        return (
            "COMMIT HISTORY:\n"
            f"{commit_history}\n"
            "\n"
            "FILE STATISTICS:\n"
            f"{input_data.diff_stats}\n"
            "\n"
            "CODE CHANGES:\n"
            f"{diff_content}"
        )

    def _build_llm_messages(self, input_data: TrainingRunInput) -> List[Dict[str, str]]:
        """
        Build the conversation messages for LLM based on training run input data.

        Args:
            input_data: TrainingRunInput containing commits, diff_stats, and diff_content

        Returns:
            List of messages for conversation
        """
        # Get few-shot examples
        examples = TRAINING_EXAMPLES

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add examples
        for example_input, expected_response in examples:
            user_message = self._format_user_message(example_input)
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": expected_response})

        # Add current request
        current_message = self._format_user_message(input_data)
        messages.append({"role": "user", "content": current_message})

        return messages

    def _build_git_summary(self, commits: List[GitCommit]) -> str:
        """
        Build a git-based summary when LLM is not available.

        Args:
            commits: List of commits in the branch

        Returns:
            Formatted git summary string

        Raises:
            ValueError: If no commits are provided
        """
        if not commits:
            raise ValueError("No commits found in this training run.")

        # Build commit history section
        commit_details = []
        for commit in commits:
            commit_details.append(f"- {commit.hash[:8]}: {commit.message} (by {commit.author} on {commit.date})")

        commit_count = len(commits)
        commit_summary = f"Training run based on {commit_count} commit{'s' if commit_count != 1 else ''}:\n\n"
        commit_summary += "\n".join(commit_details)

        return commit_summary

    def _add_uncommitted_changes_disclaimer(
        self, description: str, commit_hash: str, has_uncommitted_changes: bool
    ) -> str:
        """
        Add disclaimer about uncommitted changes if present.

        Args:
            description: The generated description
            commit_hash: The commit hash
            has_uncommitted_changes: Whether there were uncommitted changes

        Returns:
            Description with disclaimer prepended if needed
        """
        if has_uncommitted_changes:
            disclaimer = f"WARNING: Training run {commit_hash[:8]} included uncommitted changes\n\n"
            return disclaimer + description
        return description


# System prompt with context and instructions
SYSTEM_PROMPT = (
    "You are tasked with creating concise, informative descriptions for machine learning training runs.\n\n"
    "CONTEXT:\n"
    "This is a reinforcement learning project called 'Metta AI' that focuses on the emergence of cooperation "
    "and alignment in multi-agent AI systems. It creates a model organism for complex multi-agent gridworld "
    "environments to study the impact of social dynamics (like kinship and mate selection) on learning and "
    "cooperative behaviors.\n\n"
    "The codebase consists of:\n"
    "- Core Python implementation for agents, maps, RL algorithms, simulation\n"
    "- C++/Python grid environment implementation\n"
    "- Visualization and replay tools\n\n"
    "TASK:\n"
    "Produce exactly one paragraph (≤ 80 words) summarising the entire run. Never mention commit ids, hashes, "
    "file names, or headings. Focus on the scientific/experimental aspects that a researcher would care about "
    "when reviewing results - highlight the main features, experiments, or improvements that were being tested "
    "using technical language appropriate for ML researchers."
)

# Training examples for few-shot learning
TRAINING_EXAMPLES: List[tuple[TrainingRunInput, str]] = [
    # Example 1: Enhanced Progressive Curriculum (commit 748bff35)
    (
        TrainingRunInput(
            commits=[
                GitCommit(
                    hash="748bff3503d234414f9ed507762d641727529277",
                    message="Enhanced Progressive Curriculum (#1178)",
                    author="mstormbull",
                    date="2025-06-27",
                )
            ],
            diff_stats=""" .../learning_progress.yaml}                        |   0
 .../curriculum/navigation/progressive.yaml         |  24 ++++-
 configs/env/mettagrid/curriculum/progressive.yaml  |  19 ++++
 configs/user/bullm.yaml                            |  37 +++++++
 configs/user/bullm_lp.yaml                         |  36 -------
 mettagrid/src/metta/mettagrid/curriculum/core.py   |   4 +
 .../src/metta/mettagrid/curriculum/progressive.py  | 117 +++++++++++++++------
 mettagrid/src/metta/mettagrid/mettagrid_env.py     |   5 +
 8 files changed, 172 insertions(+), 70 deletions(-)""",
            diff_content="""diff --git a/configs/env/mettagrid/curriculum/navigation/progressive.yaml \\
b/configs/env/mettagrid/curriculum/navigation/progressive.yaml
index 720422717..f15291846 100644
--- a/configs/env/mettagrid/curriculum/navigation/progressive.yaml
+++ b/configs/env/mettagrid/curriculum/navigation/progressive.yaml
@@ -1,7 +1,23 @@
-_target_: metta.mettagrid.curriculum.progressive.ProgressiveMultiTaskCurriculum
+# Inherit from base progressive configuration
+defaults:
+  - /env/mettagrid/curriculum/progressive@
+  - _self_

+# Navigation-specific tasks
 tasks:
-  /env/mettagrid/navigation/training/small: 1
-  /env/mettagrid/navigation/training/medium: 1
-  /env/mettagrid/navigation/training/large: 1
   /env/mettagrid/navigation/training/terrain_from_numpy: 1
+  /env/mettagrid/navigation/training/cylinder_world: 1
+  /env/mettagrid/navigation/training/varied_terrain_sparse: 1
+  /env/mettagrid/navigation/training/varied_terrain_balanced: 1
+  /env/mettagrid/navigation/training/varied_terrain_maze: 1
+  /env/mettagrid/navigation/training/varied_terrain_dense: 1
+
+# Navigation-specific env_overrides (extends base env_overrides)
+env_overrides:
+  desync_episodes: true
+  game:
+    num_agents: 4
+
+# Navigation-specific progressive parameters (overrides base defaults)
+performance_threshold: 0.95
+progression_rate: 0.001
diff --git a/mettagrid/src/metta/mettagrid/curriculum/progressive.py \\
b/mettagrid/src/metta/mettagrid/curriculum/progressive.py
index b188e60b0..4ce2e74e2 100644
--- a/mettagrid/src/metta/mettagrid/curriculum/progressive.py
+++ b/mettagrid/src/metta/mettagrid/curriculum/progressive.py
@@ -34,56 +35,112 @@ class ProgressiveCurriculum(SamplingCurriculum):


 class ProgressiveMultiTaskCurriculum(RandomCurriculum):
-    \"\"\"Curriculum that starts with higher probabilities for earlier tasks
-    and gradually shifts to favor later tasks over time.\"\"\"
+    \"\"\"Curriculum that blends multiple tasks using gating mechanisms and advances progression based on
+    smoothed performance or time.\"\"\"

     def __init__(
         self,
         tasks: Dict[str, float],
-        env_overrides: DictConfig,
-        progression_rate: float = 0.00001,
-        initial_skew: float = 5.0,
+        env_overrides: Optional[DictConfig] = None,
+        performance_threshold: float = 0.8,
+        smoothing: float = 0.1,
+        progression_rate: float = 0.01,
+        progression_mode: str = "perf",
+        blending_smoothness: float = 0.5,
+        blending_mode: str = "logistic",
     ):
+        if env_overrides is None:
+            env_overrides = DictConfig({})
         super().__init__(tasks, env_overrides)
-        self._task_order = list(tasks.keys())  # Preserve order from dict
-        self._progression_rate = progression_rate  # How fast to shift probabilities
-        self._initial_skew = initial_skew  # How much to favor early tasks initially
+        if progression_mode not in ["time", "perf"]:
+            raise ValueError("progression_mode must be either 'time' or 'perf'")
+        if blending_mode not in ["logistic", "linear"]:
+            raise ValueError("blending_mode must be either 'logistic' or 'linear'")
+        self._task_order = list(tasks.keys())
+        self._performance_threshold = performance_threshold
+        self._smoothing = smoothing
+        self._progression_rate = progression_rate
+        self._progression_mode = progression_mode
+        self._blending_smoothness = blending_smoothness
+        self._blending_mode = blending_mode
+        self._progress = 0.0  # initialization of the progress value parameterizing the trajectory
+        self._smoothed_performance = 0.0
         self._step_count = 0
-
-        # Initialize weights heavily skewed toward beginning
+        self._last_score = None
         self._update_progressive_weights()""",
        ),
        (
            "Adds a performance-driven progressive curriculum: task weights are blended via logistic or linear gates,"
            "progression speed is tunable, and smoothed performance must reach 0.95 before advancing. Extends the"
            "navigation suite with terrain_from_numpy, cylinder_world, and varied_terrain tasks and enables"
            "desync_episodes overrides."
        ),
    ),
    # Example 2: MettaConf Multi-commit (commits 34313ae4..f4c91a71)
    (
        TrainingRunInput(
            commits=[
                GitCommit(
                    hash="34313ae4e0433a09b5eb63aa7757ea669f9c560c",
                    message="mettaconf - experimental implementation",
                    author="Vyacheslav Matyukhin",
                    date="2025-04-24",
                ),
                GitCommit(
                    hash="c4139ec3bb9377a82b808c650900643bc68ec557",
                    message="bugfix",
                    author="Vyacheslav Matyukhin",
                    date="2025-04-24",
                ),
                GitCommit(
                    hash="f4c91a71984a421d2ddac00a70247a40c2c72622",
                    message="patch nested configs",
                    author="Vyacheslav Matyukhin",
                    date="2025-04-24",
                ),
            ],
            diff_stats=""" metta/sweep/README.md                      | 149 ++++++++++
 metta/sweep/__init__.py                    |   7 +
 metta/sweep/protein.py                     | 419 +++++++++++++++++++++++++++++
 metta/sweep/protein_metta.py               |  36 +++
 metta/sweep/protein_wandb.py               | 272 +++++++++++++++++++
 metta/util/__init__.py                     |   1 +
 metta/util/numpy/__init__.py               |   5 +
 metta/util/numpy/clean_numpy_types.py      |  14 +
 pyproject.toml                             |   2 +
 tests/sweep/__init__.py                    |   1 +
 tests/sweep/test_protein_metta.py          | 197 ++++++++++++++
 tests/sweep/test_protein_wandb.py          | 273 +++++++++++++++++++
 tests/util/__init__.py                     |   1 +
 tests/util/numpy/__init__.py               |   1 +
 tests/util/numpy/test_clean_numpy_types.py | 153 +++++++++++
 uv.lock                                    |   4 +
 16 files changed, 1535 insertions(+)""",
            diff_content="""diff --git a/metta/sweep/protein.py b/metta/sweep/protein.py
new file mode 100644
index 000000000..8c9e4bb27
--- /dev/null
+++ b/metta/sweep/protein.py
@@ -0,0 +1,419 @@
+\"\"\"
+Protein: A Bayesian hyperparameter optimization system using Gaussian Processes.
+
+This module implements a sophisticated hyperparameter optimization framework with:
+- Gaussian Process models for sample-efficient search
+- Multiple parameter distribution support (log_normal, uniform, logit_normal, uniform_pow2)
+- Acquisition function optimization using Expected Improvement
+- Multi-objective optimization capabilities
+- Cost-aware optimization with budget constraints
+\"\"\"
+
+import logging
+import math
+from typing import Any, Dict, List, Optional, Tuple, Union
+
+import numpy as np
+import torch
+import torch.nn as nn
+from botorch.acquisition import ExpectedImprovement
+from botorch.models import SingleTaskGP
+from botorch.optim import optimize_acqf
+from gpytorch.kernels import ScaleKernel, RBFKernel
+from gpytorch.mlls import ExactMarginalLogLikelihood
+
+logger = logging.getLogger(__name__)
+
+
+class ParameterDistribution:
+    \"\"\"Base class for parameter distributions in hyperparameter space.\"\"\"
+
+    def __init__(self, min_val: float, max_val: float, scale: Union[str, float] = "auto"):
+        self.min_val = min_val
+        self.max_val = max_val
+        self.scale = scale
+
+    def sample(self) -> float:
+        \"\"\"Sample a value from this distribution.\"\"\"
+        raise NotImplementedError
+
+    def transform_to_unit(self, value: float) -> float:
+        \"\"\"Transform value from parameter space to [0,1] unit space.\"\"\"
+        raise NotImplementedError
+
+    def transform_from_unit(self, unit_value: float) -> float:
+        \"\"\"Transform value from [0,1] unit space to parameter space.\"\"\"
+        raise NotImplementedError
+
+
+class LogNormalDistribution(ParameterDistribution):
+    \"\"\"Log-normal distribution for parameters like learning rates.\"\"\"
+
+    def sample(self) -> float:
+        log_min = math.log(self.min_val)
+        log_max = math.log(self.max_val)
+        log_val = np.random.uniform(log_min, log_max)
+        return math.exp(log_val)
+
+    def transform_to_unit(self, value: float) -> float:
+        log_val = math.log(max(value, self.min_val))
+        log_min = math.log(self.min_val)
+        log_max = math.log(self.max_val)
+        return (log_val - log_min) / (log_max - log_min)
+
+    def transform_from_unit(self, unit_value: float) -> float:
+        log_min = math.log(self.min_val)
+        log_max = math.log(self.max_val)
+        log_val = log_min + unit_value * (log_max - log_min)
+        return math.exp(log_val)""",
        ),
        (
            "Replaces CARBS with 'Protein', a Bayesian HPO engine using single-task Gaussian Processes with RBF"
            "kernels, Expected Improvement acquisition, and multi-objective, cost-aware search. Supports flexible"
            "parameter distributions (log_normal, uniform_pow2, etc.) and streams results to WandB."
        ),
    ),
    # Example 3: Architecture change (commit ca4b293c)
    (
        TrainingRunInput(
            commits=[
                GitCommit(
                    hash="ca4b293c026521af9d0123eaaffa303bc0d34a5a",
                    message="Vicious deletion of the post-lstm relu (#1221)",
                    author="Alex Vardakostas",
                    date="2025-06-27",
                )
            ],
            diff_stats=""" configs/agent/fast.yaml              | 9 ++-------
 configs/agent/latent_attn_med.yaml   | 9 ++-------
 configs/agent/latent_attn_small.yaml | 9 ++-------
 configs/agent/latent_attn_tiny.yaml  | 9 ++-------
 configs/user/alex.yaml               | 4 ++--
 5 files changed, 10 insertions(+), 30 deletions(-)""",
            diff_content="""diff --git a/configs/agent/fast.yaml b/configs/agent/fast.yaml
index 4354c6fd0..6fee692f5 100644
--- a/configs/agent/fast.yaml
+++ b/configs/agent/fast.yaml
@@ -72,15 +72,10 @@ components:
     nn_params:
       num_layers: 2

-  core_relu:
-    _target_: metta.agent.lib.nn_layer_library.ReLU
-    sources:
-      - name: _core_
-
   critic_1:
     _target_: metta.agent.lib.nn_layer_library.Linear
     sources:
-      - name: core_relu
+      - name: _core_
     nn_params:
       out_features: 1024
     nonlinearity: nn.Tanh
@@ -97,7 +92,7 @@ components:
   actor_1:
     _target_: metta.agent.lib.nn_layer_library.Linear
     sources:
-      - name: core_relu
+      - name: _core_
     nn_params:
       out_features: 512""",
        ),
        (
            "Deletes the intermediate 'core_relu' so LSTM outputs feed critic and actor linear layers directly across"
            "agents, testing whether trimming activations improves gradient flow and sample efficiency."
        ),
    ),
]
