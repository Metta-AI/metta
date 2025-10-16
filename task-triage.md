**1. 30 sweeps per confirmed result (gid 1211414008323007)**
Difficulty: Very High
Likely entails scripting experiment configs and automation for large sweep runs. This will touch experiments/ configs plus tools/run.py orchestration. Expect to add regression tests and run uv run pytest before closing.

**2. Benchmark all architectures (gid 1211501032736057)**
Difficulty: Very High
Likely entails running controlled comparisons and capturing metrics. Expect to add regression tests and run uv run pytest before closing.

**3. Generational sweeps (gid 1211366023715604)**
Difficulty: Very High
Likely entails scripting experiment configs and automation for large sweep runs. This will touch experiments/ configs plus tools/run.py orchestration. Expect to add regression tests and run uv run pytest before closing.

**4. Implement automated early termination policy (waiting for Skypilot) (gid 1210768867764961)**
Difficulty: Very High
Likely entails implementing new core logic and validating it end-to-end. This will touch the scripts/skypilot utilities and cluster orchestration helpers, packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**5. Define scaling experiment recipe and analysis pipeline (gid 1210768867894504)**
Difficulty: High
Likely entails producing reproducible analysis along with documentation, and running exploratory experiments and interpreting results. This will touch analysis notebooks under docs/ or experiments/analysis. Expect to add regression tests and run uv run pytest before closing.

**6. Design simple but effective scripted NPCs (gid 1210943495769120)**
Difficulty: High
Likely entails expanding NPC assets and verifying behavior in simulation. This will touch packages/cogames/npc and evaluation pipelines. Expect to add regression tests and run uv run pytest before closing.

**7. Evolutionary Policy Optimization (gid 1210578513824126)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**8. FACTS: A Factored State-Space Framework for World Modelling (gid 1211413398528738)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**9. Fix policy_store._cached_prs (gid 1209148137213831)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**10. Goal: Establish SOTA results in arena with kickstarting (gid 1210803372736877)**
Difficulty: High
Likely entails coordinating teacher/student training recipes and validating performance. This will touch experiments/recipes/kickstarting assets. Plan to capture notes in docs/ or the appropriate README.

**11. Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents (gid 1209070474559707)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch agent/ runtime and packages/cogames/policies, packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**12. Integrate NPC through losses post-dehydration (gid 1211365428128399)**
Difficulty: High
Likely entails aligning interfaces between subsystems and updating tests, and touching the loss stack and ensuring gradients stay stable, and expanding NPC assets and verifying behavior in simulation. This will touch packages/cogames/losses and training loops in packages/common/training, packages/cogames/npc and evaluation pipelines. Expect to add regression tests and run uv run pytest before closing.

**13. Memory Architecture, memory tokens? (gid 1209661379538190)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**14. Modal: High-performance AI infrastructure (gid 1209480502135241)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**15. Multi-Policy Training (PufferLib) (gid 1209017937764257)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**16. Research: Can we train a policy to help a uri specified NPC get more reward? (gid 1210979764834655)**
Difficulty: High
Likely entails expanding NPC assets and verifying behavior in simulation. This will touch packages/cogames/npc and evaluation pipelines, packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**17. Sweep over learning progress hypers (gid 1210768737095586)**
Difficulty: High
Likely entails scripting experiment configs and automation for large sweep runs. This will touch experiments/ configs plus tools/run.py orchestration, experiments/ recipes and config files. Expect to add regression tests and run uv run pytest before closing.

**18. Train a transformer on hard envs (gid 1210284652865440)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies/transformers and shared model utilities. Expect to add regression tests and run uv run pytest before closing.

**19. Training against a pre-existing policy (gid 1211366023380595)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies. Expect to add regression tests and run uv run pytest before closing.

**20. Training Pipeline (gid 1211366079632345)**
Difficulty: High
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**21. Automatically control update_epochs (gid 1210797922141228)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**22. Clean interfaces to alternative training systems (gid 1210794035717156)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**23. Clean up curricula <> simulations interactions. tools should accept curricula but simulations should not (gid 1209843025820874)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/curricula. Plan to capture notes in docs/ or the appropriate README.

**24. Cost/Resource sweeping (Axel) (gid 1211366025760607)**
Difficulty: Medium-High
Likely entails scripting experiment configs and automation for large sweep runs. This will touch experiments/ configs plus tools/run.py orchestration. Expect to add regression tests and run uv run pytest before closing.

**25. Debugging Reinforcement Learning Systems (gid 1209805021772139)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**26. Demonstrate kickstarting efficiency (gid 1210768737314651)**
Difficulty: Medium-High
Likely entails coordinating teacher/student training recipes and validating performance. This will touch experiments/recipes/kickstarting assets. Plan to capture notes in docs/ or the appropriate README.

**27. Diagnose and repair LSTM in cogames (gid 1211643546103111)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies/recurrent models. Plan to capture notes in docs/ or the appropriate README.

**28. Dreamer v3 (gid 1211366024221512)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**29. Early stopping training (gid 1209208922660706)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**30. Evals broken with NPC policies (gid 1210786852138807)**
Difficulty: Medium-High
Likely entails expanding NPC assets and verifying behavior in simulation. This will touch packages/cogames/npc and evaluation pipelines. Expect to add regression tests and run uv run pytest before closing.

**31. Experiment with diversity (DIAYN) (gid 1211366169326504)**
Difficulty: Medium-High
Likely entails running exploratory experiments and interpreting results. Expect to add regression tests and run uv run pytest before closing.

**32. Experiment with error-based signals, e.g. regret (gid 1210769219224464)**
Difficulty: Medium-High
Likely entails running exploratory experiments and interpreting results. Expect to add regression tests and run uv run pytest before closing.

**33. Experiment with fitnotyping (gid 1211377548529720)**
Difficulty: Medium-High
Likely entails running exploratory experiments and interpreting results. Expect to add regression tests and run uv run pytest before closing.

**34. Generate replays during training (gid 1209041170403516)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/replay and logging. Expect to add regression tests and run uv run pytest before closing.

**35. Goal: generate a stable Pattern for running for Performance Evaluations/comparisons (gid 1210793382718996)**
Difficulty: Medium-High
Likely entails driving a multi-step effort that likely spans design, implementation, and validation. This will touch packages/cogames/eval pipelines. Plan to capture notes in docs/ or the appropriate README.

**36. Imitation learning (gid 1211366168188140)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**37. Implement Continuous Dropout (gid 1210282198057739)**
Difficulty: Medium-High
Likely entails implementing new core logic and validating it end-to-end. Expect to add regression tests and run uv run pytest before closing.

**38. Implement Nevergrad adapter (gid 1211034485747674)**
Difficulty: Medium-High
Likely entails implementing new core logic and validating it end-to-end. Expect to add regression tests and run uv run pytest before closing.

**39. Integrate noisy linear (gid 1210286336916414)**
Difficulty: Medium-High
Likely entails aligning interfaces between subsystems and updating tests. Expect to add regression tests and run uv run pytest before closing.

**40. Kickstarting single student from single teacher (gid 1209661379538164)**
Difficulty: Medium-High
Likely entails coordinating teacher/student training recipes and validating performance. This will touch experiments/recipes/kickstarting assets. Plan to capture notes in docs/ or the appropriate README.

**41. LSTM Entropy (gid 1209548780079332)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies/recurrent models. Plan to capture notes in docs/ or the appropriate README.

**42. Make scorecards for comparing performance of different training regimes (gid 1210769218369227)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**43. Measure Learning Progress performance in arena and navigation against other curricula (gid 1209843025820875)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/curricula. Plan to capture notes in docs/ or the appropriate README.

**44. MettaGrid training on Google Colab (gid 1209208922660670)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**45. Mining replays for supervised learning dataset (gid 1211366081690070)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch data ingestion scripts under tools/data, packages/cogames/replay and logging. Plan to capture notes in docs/ or the appropriate README.

**46. Share a vecenv in SimulationSuite (gid 1210189468261556)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/envs and runtime wrappers in packages/common/simulation, packages/cogames/simulationsuite and related env adapters. Plan to capture notes in docs/ or the appropriate README.

**47. Simplified Temporal Consistency Reinforcement Learning (gid 1210992256238856)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**48. Spectral norm LSTM gate weights (gid 1211014686946592)**
Difficulty: Medium-High
Likely entails producing design notes and aligning stakeholders. This will touch packages/cogames/policies/recurrent models. Plan to capture notes in docs/ or the appropriate README.

**49. Supervised learning (gid 1211366166789802)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**50. Sweep over Navigation recipe (gid 1210835256607814)**
Difficulty: Medium-High
Likely entails scripting experiment configs and automation for large sweep runs. This will touch experiments/ configs plus tools/run.py orchestration. Expect to add regression tests and run uv run pytest before closing.

**51. We should post and update when hyperparams update (gid 1211366025214105)**
Difficulty: Medium-High
Likely entails making targeted updates within the training stack and verifying with tests. This will touch experiments/ recipes and config files. Plan to capture notes in docs/ or the appropriate README.

**52. Write a validation experiment for best run/hyper selection at the end of a sweep. (gid 1211366080964632)**
Difficulty: Medium-High
Likely entails scripting experiment configs and automation for large sweep runs, and running exploratory experiments and interpreting results. This will touch experiments/ configs plus tools/run.py orchestration, experiments/ recipes and config files. Expect to add regression tests and run uv run pytest before closing.

**53. AMAGO (gid 1209661379538166)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**54. Atomic saves (gid 1209394151090423)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**55. AWS instance comparison report (gid 1210768672358004)**
Difficulty: Medium
Likely entails producing reproducible analysis along with documentation, and updating infrastructure scripts and verifying provisioning end-to-end. This will touch devops/aws automation, docs/ and analyses/notes. Plan to capture notes in docs/ or the appropriate README.

**56. Chat with labmates about useful RL tricks (gid 1211414008917193)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**57. Clear design document for agent refactor (gid 1210769258927270)**
Difficulty: Medium
Likely entails driving a multi-step effort that likely spans design, implementation, and validation. This will touch agent/ runtime and packages/cogames/policies. Plan to capture notes in docs/ or the appropriate README.

**58. Continuous distillation (gid 1211366022781592)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**59. Deep Complex Networks (gid 1211419372602429)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**60. Deterministic runs (gid 1211665269726547)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**61. Doxascope (gid 1211366171266384)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**62. Easy job log access for skypilot jobs (gid 1209694588206333)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. This will touch the scripts/skypilot utilities and cluster orchestration helpers. Plan to capture notes in docs/ or the appropriate README.

**63. Empirical-precision-weighted kalman updates (gid 1209070474559682)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**64. Exploration (gid 1209661379538184)**
Difficulty: Medium
Likely entails running exploratory experiments and interpreting results. Plan to capture notes in docs/ or the appropriate README.

**65. Exploration: Deep Laplacian Representation (gid 1211414008323018)**
Difficulty: Medium
Likely entails running exploratory experiments and interpreting results. Plan to capture notes in docs/ or the appropriate README.

**66. Exploration: Successor Features/Representation (gid 1211414008323013)**
Difficulty: Medium
Likely entails running exploratory experiments and interpreting results. Plan to capture notes in docs/ or the appropriate README.

**67. Exploring Distributional RL (gid 1209734926321017)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**68. Exploring MAPG (gid 1209734926321013)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**69. External obs can support tensor or tensor dict (gid 1211366079505398)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**70. Fix WANDB timeouts in aws runs (gid 1209300681992284)**
Difficulty: Medium
Likely entails updating infrastructure scripts and verifying provisioning end-to-end. This will touch metrics logging hooks under tools/metrics and packages/cogames/logging, devops/aws automation. Plan to capture notes in docs/ or the appropriate README.

**71. GBPE (gid 1211414008323001)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**72. Give users simple access to cost information (waiting for skypilot) (gid 1210836493505871)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. This will touch the scripts/skypilot utilities and cluster orchestration helpers. Plan to capture notes in docs/ or the appropriate README.

**73. Global Observations (gid 1209661381715536)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**74. GRPO (gid 1210921009975326)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**75. Hierchiechal RL (gid 1209661381715554)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**76. HRM (Hierarchical Reasoning Model) (gid 1211334022894072)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**77. HuggingFaceTB/SmolLM2-135M · Hugging Face (gid 1211068828541795)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**78. Improve PROTEIN: (gid 1211034485747675)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**79. Investigate mixtures of experts (gid 1211386207776011)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**80. Lambda Returns (gid 1211414008323010)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**81. Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks (gid 1209148136958987)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**82. LoRA (gid 1211614906026722)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**83. Merge recommendation for default configuration for Arena (gid 1210768673357064)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**84. Mettaboxes (gid 1210397100757356)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**85. Neural Damage (gid 1209661381715534)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**86. Observation Prediction (gid 1209661381715532)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**87. Predictive Loss (gid 1210392276593852)**
Difficulty: Medium
Likely entails touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**88. Recurrent Trace Unit (gid 1211366082402133)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**89. Remove that Square Root: A New Efficient
Scale-Invariant Version of AdaGrad (gid 1209021128429249)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**90. Replicate Ada (gid 1211366169251945)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**91. Robust-to-new-versions Agent (gid 1209661381715538)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. This will touch agent/ runtime and packages/cogames/policies. Plan to capture notes in docs/ or the appropriate README.

**92. Syllabus Support (gid 1209043561406868)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**93. Test an embedded LSTM (gid 1210286336916416)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/cogames/policies/recurrent models. Plan to capture notes in docs/ or the appropriate README.

**94. Test out Muon optimizer. (gid 1209208922660224)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. This will touch packages/common/optimizers, packages/common/optimizers. Expect to add regression tests and run uv run pytest before closing.

**95. Time the various GPUs, like H100 vs. A100 vs. etc (gid 1210892820911818)**
Difficulty: Medium
Likely entails updating infrastructure scripts and verifying provisioning end-to-end. This will touch devops/hardware provisioning scripts. Plan to capture notes in docs/ or the appropriate README.

**96. Try KL loss in PPO (gid 1209208922660336)**
Difficulty: Medium
Likely entails touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**97. Try out fibrations (gid 1210527330325140)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**98. Try using average-reward and no discounting (gid 1209394151090417)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**99. v1 figure of merit formula for arena environment (gid 1210768867477704)**
Difficulty: Medium
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**100. Adam on Local Time Addressing Nonstationarity in RL with Relative Adam Timesteps (gid 1209017937764338)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. Expect to add regression tests and run uv run pytest before closing.

**101. Add CMPO (gid 1211471933325968)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**102. Add Contrastive Loss (gid 1211366081045969)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths, and touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**103. Add dormant neurons graph to Wandb / evals (gid 1209208922660846)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. This will touch metrics logging hooks under tools/metrics and packages/cogames/logging. Expect to add regression tests and run uv run pytest before closing.

**104. Add Dynamics Model: Next action prediction (gid 1211366082087987)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**105. Add Dynamics Model: Next value prediction (gid 1211377548529684)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**106. Add EMA of Future Latent State Prediction Loss (gid 1211366023633219)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths, and touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**107. Add Loss scheduling, i.e., saving the state of losses (gid 1211366081860048)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths, and touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**108. Add measurements for fairness during evaluation (gid 1209164338788423)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. This will touch packages/cogames/eval pipelines. Expect to add regression tests and run uv run pytest before closing.

**109. Add MUESLI (gid 1211366024409404)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**110. Add noise to RNN (gid 1209778674231666)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**111. Add sliding window flash attention to cortex (gid 1211590894831394)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**112. Add Stable Latent State Loss (gid 1211366082780014)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths, and touching the loss stack and ensuring gradients stay stable. This will touch packages/cogames/losses and training loops in packages/common/training. Expect to add regression tests and run uv run pytest before closing.

**113. Add unit test that same seed gives us the same model weights + starting config (gid 1211665269726540)**
Difficulty: Low
Likely entails adding a focused capability to existing codepaths. Expect to add regression tests and run uv run pytest before closing.

**114. Align with researcher tools on notebooks/plotly dashboards (gid 1211069421664747)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**115. Check VAST for 4090/5090 Dev boxes (gid 1210892820911816)**
Difficulty: Low
Likely entails updating infrastructure scripts and verifying provisioning end-to-end. This will touch infrastructure provisioning scripts in devops/. Plan to capture notes in docs/ or the appropriate README.

**116. Cogweb Agent Bucket for agents (gid 1210851220510512)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. This will touch agent/ runtime and packages/cogames/policies. Plan to capture notes in docs/ or the appropriate README.

**117. Rename EnvironmentMetaData to GameRules (gid 1211496531542715)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**118. Specify envs for replays (gid 1209208922660615)**
Difficulty: Low
Likely entails producing design notes and aligning stakeholders. This will touch packages/cogames/replay and logging. Plan to capture notes in docs/ or the appropriate README.

**119. Task Graph Test (gid 1209208922660663)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.

**120. The Road Less Scheduled (gid 1209017937764336)**
Difficulty: Low
Likely entails making targeted updates within the training stack and verifying with tests. Plan to capture notes in docs/ or the appropriate README.
