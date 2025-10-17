**Completed Tasks**

- Specify envs for replays (gid 1209208922660615) — implemented on branch `richard-replays-spec`; added evaluator configuration support for selecting environments when generating replays.
- Add measurements for fairness during evaluation (gid 1209164338788423) — implemented on branch `richard-fairness-metrics`; records group-level fairness metrics and surfaces them in evaluation dashboards.
- Deterministic runs (gid 1211665269726547) — implemented on branch `richard-deterministic-runs`; adds a deterministic mode to TrainTool that forces torch deterministic backends and disables episode desynchronization in curricula.
- Generate replays during training (gid 1209041170403516) — implemented on branch `richard-training-replays`; adds configurable replay recording for the training environment with per-environment storage.

---

**1. 30 sweeps per confirmed result (gid 1211414008323007)**
Difficulty: Very High
Assignee: Subhojeet Pramanik
Likely entails scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**2. Benchmark all architectures (gid 1211501032736057)**
Difficulty: Very High
Assignee: Tasha Pais
Likely entails running controlled comparisons and capturing metrics. Capture before/after metrics in docs/experiments and attach to the PR.

**3. Cost/Resource sweeping (Axel) (gid 1211366025760607)**
Difficulty: Very High
Assignee: Axel Kerbec
Likely entails scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**4. Generational sweeps (gid 1211366023715604)**
Difficulty: Very High
Assignee: Unassigned
Likely entails scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**5. Implement automated early termination policy (waiting for Skypilot) (gid 1210768867764961)**
Difficulty: Very High
Assignee: Dominik Farr
Likely entails implementing new core logic and validating end-to-end, and adjusting policy modules and confirming evaluation metrics while touching scripts/skypilot and orchestration tooling, packages/cogames/policies. Add regression coverage in packages/cogames/tests and run uv run pytest.

**6. Sweep over learning progress hypers (gid 1210768737095586)**
Difficulty: Very High
Assignee: Jack Heart
Likely entails scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation, experiments/recipes hyperparameter configs. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**7. Sweep over Navigation recipe (gid 1210835256607814)**
Difficulty: Very High
Assignee: Axel Kerbec
Likely entails scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**8. Define scaling experiment recipe and analysis pipeline (gid 1210768867894504)**
Difficulty: High
Assignee: Axel Kerbec
Likely entails running exploratory experiments and interpreting the results, and producing reproducible analysis and documenting findings. Store notebooks or summary tables in docs/analysis for review.

**9. Design simple but effective scripted NPCs (gid 1210943495769120)**
Difficulty: High
Assignee: Unassigned
Likely entails producing design docs to align stakeholders, and expanding NPC behaviours and verifying simulation outcomes while touching packages/cogames/npc assets and eval pipelines. Record short replay snippets to confirm NPC behaviour shifts.

**10. Evolutionary Policy Optimization (gid 1210578513824126)**
Difficulty: High
Assignee: Unassigned
Likely entails adjusting policy modules and confirming evaluation metrics while touching packages/cogames/policies. Add regression coverage in packages/cogames/tests and run uv run pytest.

**11. FACTS: A Factored State-Space Framework for World Modelling (gid 1211413398528738)**
Difficulty: High
Assignee: Emmett Shear
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**12. Fix policy_store._cached_prs (gid 1209148137213831)**
Difficulty: Obsolete
Assignee: Michael Hollander
Status: No action needed — PolicyStore was removed in commit 5eb9839a3a, eliminating _cached_prs.

**13. Goal: Establish SOTA results in arena with kickstarting (gid 1210803372736877)**
Difficulty: High
Assignee: Alexandros Vardakostas
Likely entails driving a multi-step initiative from design through validation while touching kickstarting recipes under experiments/recipes/kickstarting. Plan to run uv run pytest and capture key metrics before marking the task complete.

**14. Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents (gid 1209070474559707)**
Difficulty: High
Assignee: Emmett Shear
Likely entails adjusting policy modules and confirming evaluation metrics while touching packages/cogames/policies. Add regression coverage in packages/cogames/tests and run uv run pytest.

**15. Integrate NPC through losses post-dehydration (gid 1211365428128399)**
Difficulty: High
Assignee: Unassigned
Likely entails aligning interfaces between subsystems, and expanding NPC behaviours and verifying simulation outcomes, and touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops, packages/cogames/npc assets and eval pipelines. Record short replay snippets to confirm NPC behaviour shifts.

**16. Memory Architecture, memory tokens? (gid 1209661379538190)**
Difficulty: High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**17. Mining replays for supervised learning dataset (gid 1211366081690070)**
Difficulty: High
Assignee: Teo Ionita
Likely entails tuning replay generation and viewer integration while touching data ingestion utilities in tools/data, metta/tools/replay and metta/sim/replay_log_renderer. Generate a sample replay and ensure MettaScope can load it.

**18. Modal: High-performance AI infrastructure (gid 1209480502135241)**
Difficulty: High
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**19. Multi-Policy Training (PufferLib) (gid 1209017937764257)**
Difficulty: High
Assignee: Unassigned
Likely entails running multi-seed training loops to gather evidence, and adjusting policy modules and confirming evaluation metrics while touching packages/cogames/policies. Add regression coverage in packages/cogames/tests and run uv run pytest.

**20. Research: Can we train a policy to help a uri specified NPC get more reward? (gid 1210979764834655)**
Difficulty: High
Assignee: Unassigned
Likely entails running multi-seed training loops to gather evidence, and expanding NPC behaviours and verifying simulation outcomes, and adjusting policy modules and confirming evaluation metrics while touching packages/cogames/npc assets and eval pipelines, packages/cogames/policies. Record short replay snippets to confirm NPC behaviour shifts.

**21. Train a transformer on hard envs (gid 1210284652865440)**
Difficulty: High
Assignee: Richard Higgins
Likely entails running multi-seed training loops to gather evidence while touching packages/cogames/policies/transformers and shared model utils, mettagrid/config definitions and curriculum tasks. Plan to run uv run pytest and capture key metrics before marking the task complete.

**22. Training against a pre-existing policy (gid 1211366023380595)**
Difficulty: High
Assignee: Unassigned
Likely entails running multi-seed training loops to gather evidence, and adjusting policy modules and confirming evaluation metrics while touching packages/cogames/policies. Add regression coverage in packages/cogames/tests and run uv run pytest.

**23. Training Pipeline (gid 1211366079632345)**
Difficulty: High
Assignee: Unassigned
Likely entails running multi-seed training loops to gather evidence. Plan to run uv run pytest and capture key metrics before marking the task complete.

**24. Write a validation experiment for best run/hyper selection at the end of a sweep. (gid 1211366080964632)**
Difficulty: High
Assignee: Axel Kerbec
Likely entails running exploratory experiments and interpreting the results, and documenting behaviour and tightening tests, and scripting large parameter sweeps with automation while touching experiments/recipes and tools/run.py automation, experiments/recipes hyperparameter configs. Plan to sanity-check with a smoke sweep via tools/run.py and record outcomes.

**25. Automatically control update_epochs (gid 1210797922141228)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**26. Clean interfaces to alternative training systems (gid 1210794035717156)**
Difficulty: Medium-High
Assignee: Matthew Bull
Likely entails running multi-seed training loops to gather evidence. Plan to run uv run pytest and capture key metrics before marking the task complete.

**27. Clean up curricula <> simulations interactions. tools should accept curricula but simulations should not (gid 1209843025820874)**
Difficulty: Medium-High
Assignee: Michael Hollander
Likely entails making targeted updates within the training stack and verifying behaviour while touching metta/sim and packages/mettagrid. Plan to run uv run pytest and capture key metrics before marking the task complete.

**28. Debugging Reinforcement Learning Systems (gid 1209805021772139)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**29. Demonstrate kickstarting efficiency (gid 1210768737314651)**
Difficulty: Medium-High
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour while touching kickstarting recipes under experiments/recipes/kickstarting. Plan to run uv run pytest and capture key metrics before marking the task complete.

**30. Diagnose and repair LSTM in cogames (gid 1211643546103111)**
Difficulty: Medium-High
Assignee: Dominik Farr
Likely entails making targeted updates within the training stack and verifying behaviour while touching packages/cogames/policies/recurrent. Plan to run uv run pytest and capture key metrics before marking the task complete.

**31. Dreamer v3 (gid 1211366024221512)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**32. Early stopping training (gid 1209208922660706)**
Difficulty: Medium-High
Assignee: Dominik Farr
Likely entails running multi-seed training loops to gather evidence. Plan to run uv run pytest and capture key metrics before marking the task complete.

**33. Evals broken with NPC policies (gid 1210786852138807)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails expanding NPC behaviours and verifying simulation outcomes while touching packages/cogames/npc assets and eval pipelines. Record short replay snippets to confirm NPC behaviour shifts.

**34. Experiment with diversity (DIAYN) (gid 1211366169326504)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails running exploratory experiments and interpreting the results. Plan to run uv run pytest and capture key metrics before marking the task complete.

**35. Experiment with error-based signals, e.g. regret (gid 1210769219224464)**
Difficulty: Medium-High
Assignee: Jack Heart
Likely entails running exploratory experiments and interpreting the results. Plan to run uv run pytest and capture key metrics before marking the task complete.

**36. Experiment with fitnotyping (gid 1211377548529720)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails running exploratory experiments and interpreting the results. Plan to run uv run pytest and capture key metrics before marking the task complete.

**37. Goal: generate a stable Pattern for running for Performance Evaluations/comparisons (gid 1210793382718996)**
Difficulty: Medium-High
Assignee: Matthew Bull
Likely entails driving a multi-step initiative from design through validation. Plan to run uv run pytest and capture key metrics before marking the task complete.

**38. Imitation learning (gid 1211366168188140)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**39. Implement Continuous Dropout (gid 1210282198057739)**
Difficulty: Medium-High
Assignee: harshbhatt7585@gmail.com
Likely entails implementing new core logic and validating end-to-end. Plan to run uv run pytest and capture key metrics before marking the task complete.

**40. Implement Nevergrad adapter (gid 1211034485747674)**
Difficulty: Medium-High
Assignee: Axel Kerbec
Likely entails implementing new core logic and validating end-to-end. Plan to run uv run pytest and capture key metrics before marking the task complete.

**41. Integrate noisy linear (gid 1210286336916414)**
Difficulty: Medium-High
Assignee: Alexandros Vardakostas
Likely entails aligning interfaces between subsystems. Plan to run uv run pytest and capture key metrics before marking the task complete.

**42. Kickstarting single student from single teacher (gid 1209661379538164)**
Difficulty: Medium-High
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour while touching kickstarting recipes under experiments/recipes/kickstarting. Plan to run uv run pytest and capture key metrics before marking the task complete.

**43. LSTM Entropy (gid 1209548780079332)**
Difficulty: Medium-High
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour while touching packages/cogames/policies/recurrent. Plan to run uv run pytest and capture key metrics before marking the task complete.

**44. Make scorecards for comparing performance of different training regimes (gid 1210769218369227)**
Difficulty: Medium-High
Assignee: Axel Kerbec
Likely entails running multi-seed training loops to gather evidence. Plan to run uv run pytest and capture key metrics before marking the task complete.

**45. Measure Learning Progress performance in arena and navigation against other curricula (gid 1209843025820875)**
Difficulty: Medium-High
Assignee: Jack Heart
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**46. MettaGrid training on Google Colab (gid 1209208922660670)**
Difficulty: Medium-High
Assignee: harshbhatt7585@gmail.com
Likely entails running multi-seed training loops to gather evidence. Plan to run uv run pytest and capture key metrics before marking the task complete.

**47. Share a vecenv in SimulationSuite (gid 1210189468261556)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour while touching metta/rl/vecenv and related simulation wrappers, metta/sim and packages/mettagrid, mettagrid/config definitions and curriculum tasks. Plan to run uv run pytest and capture key metrics before marking the task complete.

**48. Simplified Temporal Consistency Reinforcement Learning (gid 1210992256238856)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**49. Spectral norm LSTM gate weights (gid 1211014686946592)**
Difficulty: Medium-High
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour while touching packages/cogames/policies/recurrent. Plan to run uv run pytest and capture key metrics before marking the task complete.

**50. Supervised learning (gid 1211366166789802)**
Difficulty: Medium-High
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**51. v1 figure of merit formula for arena environment (gid 1210768867477704)**
Difficulty: Medium-High
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour while touching mettagrid/config definitions and curriculum tasks. Plan to run uv run pytest and capture key metrics before marking the task complete.

**52. We should post and update when hyperparams update (gid 1211366025214105)**
Difficulty: Medium-High
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour while touching experiments/recipes hyperparameter configs. Plan to run uv run pytest and capture key metrics before marking the task complete.

**53. AMAGO (gid 1209661379538166)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**54. Atomic saves (gid 1209394151090423)**
Difficulty: Medium
Assignee: Michael Hollander
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**55. AWS instance comparison report (gid 1210768672358004)**
Difficulty: Medium
Assignee: Axel Kerbec
Likely entails producing reproducible analysis and documenting findings while touching devops/aws automation. Store notebooks or summary tables in docs/analysis for review.

**56. Chat with labmates about useful RL tricks (gid 1211414008917193)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**57. Clear design document for agent refactor (gid 1210769258927270)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails producing design docs to align stakeholders. Plan to run uv run pytest and capture key metrics before marking the task complete.

**58. Continuous distillation (gid 1211366022781592)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**59. Deep Complex Networks (gid 1211419372602429)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**60. Doxascope (gid 1211366171266384)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**61. Easy job log access for skypilot jobs (gid 1209694588206333)**
Difficulty: Medium
Assignee: Teo Ionita
Likely entails making targeted updates within the training stack and verifying behaviour while touching scripts/skypilot and orchestration tooling. Plan to run uv run pytest and capture key metrics before marking the task complete.

**62. Empirical-precision-weighted kalman updates (gid 1209070474559682)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**63. Exploration (gid 1209661379538184)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**64. Exploration: Deep Laplacian Representation (gid 1211414008323018)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**65. Exploration: Successor Features/Representation (gid 1211414008323013)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**66. Exploring Distributional RL (gid 1209734926321017)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**67. Exploring MAPG (gid 1209734926321013)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**68. External obs can support tensor or tensor dict (gid 1211366079505398)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**69. Fix WANDB timeouts in aws runs (gid 1209300681992284)**
Difficulty: Medium
Assignee: Teo Ionita
Likely entails making targeted updates within the training stack and verifying behaviour while touching logging hooks under metta/common/wandb and analysis notebooks, devops/aws automation. Plan to run uv run pytest and capture key metrics before marking the task complete.

**70. GBPE (gid 1211414008323001)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**71. Give users simple access to cost information (waiting for skypilot) (gid 1210836493505871)**
Difficulty: Medium
Assignee: Dominik Farr
Likely entails making targeted updates within the training stack and verifying behaviour while touching scripts/skypilot and orchestration tooling. Plan to run uv run pytest and capture key metrics before marking the task complete.

**72. Global Observations (gid 1209661381715536)**
Difficulty: Medium
Assignee: Robb Walters
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**73. GRPO (gid 1210921009975326)**
Difficulty: Medium
Assignee: harshbhatt7585@gmail.com
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**74. Hierchiechal RL (gid 1209661381715554)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**75. HRM (Hierarchical Reasoning Model) (gid 1211334022894072)**
Difficulty: Medium
Assignee: harshbhatt7585@gmail.com
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**76. HuggingFaceTB/SmolLM2-135M · Hugging Face (gid 1211068828541795)**
Difficulty: Medium
Assignee: Richard Higgins
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**77. Improve PROTEIN: (gid 1211034485747675)**
Difficulty: Medium
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**78. Investigate mixtures of experts (gid 1211386207776011)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**79. Lambda Returns (gid 1211414008323010)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**80. Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks (gid 1209148136958987)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**81. LoRA (gid 1211614906026722)**
Difficulty: Medium
Assignee: Richard Higgins
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**82. Merge recommendation for default configuration for Arena (gid 1210768673357064)**
Difficulty: Medium
Assignee: Tasha Pais
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**83. Mettaboxes (gid 1210397100757356)**
Difficulty: Medium
Assignee: Richard Higgins
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**84. Neural Damage (gid 1209661381715534)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**85. Observation Prediction (gid 1209661381715532)**
Difficulty: Medium
Assignee: Tasha Pais
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**86. Predictive Loss (gid 1210392276593852)**
Difficulty: Medium
Assignee: Tasha Pais
Likely entails touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**87. Recurrent Trace Unit (gid 1211366082402133)**
Difficulty: Medium
Assignee: Subhojeet Pramanik
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**88. Remove that Square Root: A New Efficient
Scale-Invariant Version of AdaGrad (gid 1209021128429249)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**89. Replicate Ada (gid 1211366169251945)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**90. Robust-to-new-versions Agent (gid 1209661381715538)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**91. Syllabus Support (gid 1209043561406868)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**92. Test an embedded LSTM (gid 1210286336916416)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour while touching packages/cogames/policies/recurrent. Plan to run uv run pytest and capture key metrics before marking the task complete.

**93. Test out Muon optimizer. (gid 1209208922660224)**
Difficulty: Medium
Assignee: harshbhatt7585@gmail.com
Likely entails making targeted updates within the training stack and verifying behaviour while touching packages/common/optimizers. Plan to run uv run pytest and capture key metrics before marking the task complete.

**94. Time the various GPUs, like H100 vs. A100 vs. etc (gid 1210892820911818)**
Difficulty: Medium
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour while touching devops/hardware provisioning scripts. Plan to run uv run pytest and capture key metrics before marking the task complete.

**95. Try KL loss in PPO (gid 1209208922660336)**
Difficulty: Medium
Assignee: harshbhatt7585@gmail.com
Likely entails touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**96. Try out fibrations (gid 1210527330325140)**
Difficulty: Medium
Assignee: Alexandros Vardakostas
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**97. Try using average-reward and no discounting (gid 1209394151090417)**
Difficulty: Medium
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**98. Adam on Local Time Addressing Nonstationarity in RL with Relative Adam Timesteps (gid 1209017937764338)**
Difficulty: Low
Assignee: Unassigned
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**99. Add CMPO (gid 1211471933325968)**
Difficulty: Low
Assignee: Unassigned
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**100. Add Contrastive Loss (gid 1211366081045969)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules, and touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**101. Add dormant neurons graph to Wandb / evals (gid 1209208922660846)**
Difficulty: Low
Assignee: Michael Hollander
Likely entails adding focused functionality to existing modules while touching logging hooks under metta/common/wandb and analysis notebooks. Plan to run uv run pytest and capture key metrics before marking the task complete.

**102. Add Dynamics Model: Next action prediction (gid 1211366082087987)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**103. Add Dynamics Model: Next value prediction (gid 1211377548529684)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**104. Add EMA of Future Latent State Prediction Loss (gid 1211366023633219)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules, and touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**105. Add Loss scheduling, i.e., saving the state of losses (gid 1211366081860048)**
Difficulty: Low
Assignee: Alexandros Vardakostas
Likely entails adding focused functionality to existing modules, and touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**106. Add MUESLI (gid 1211366024409404)**
Difficulty: Low
Assignee: Alexandros Vardakostas
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**107. Add noise to RNN (gid 1209778674231666)**
Difficulty: Low
Assignee: Alexandros Vardakostas
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**108. Add sliding window flash attention to cortex (gid 1211590894831394)**
Difficulty: Low
Assignee: Subhojeet Pramanik
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**109. Add Stable Latent State Loss (gid 1211366082780014)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules, and touching the loss stack and keeping gradients stable while touching packages/cogames/losses plus metta/rl/training loops. Add focused unit tests around loss computation and run uv run pytest.

**110. Add unit test that same seed gives us the same model weights + starting config (gid 1211665269726540)**
Difficulty: Low
Assignee: Tasha Pais
Likely entails adding focused functionality to existing modules. Plan to run uv run pytest and capture key metrics before marking the task complete.

**111. Align with researcher tools on notebooks/plotly dashboards (gid 1211069421664747)**
Difficulty: Low
Assignee: Matthew Bull
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**112. Check VAST for 4090/5090 Dev boxes (gid 1210892820911816)**
Difficulty: Low
Assignee: Axel Kerbec
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**113. Cogweb Agent Bucket for agents (gid 1210851220510512)**
Difficulty: Low
Assignee: Richard Higgins
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

**114. Rename EnvironmentMetaData to GameRules (gid 1211496531542715)**
Difficulty: Low
Assignee: Michael Hollander
Likely entails performing schema and naming clean-up while touching mettagrid/config definitions and curriculum tasks. Plan to run uv run pytest and capture key metrics before marking the task complete.

**115. Task Graph Test (gid 1209208922660663)**
Difficulty: Low
Assignee: Matthew Bull
Likely entails making targeted updates within the training stack and verifying behaviour. Plan to run uv run pytest and capture key metrics before marking the task complete.

