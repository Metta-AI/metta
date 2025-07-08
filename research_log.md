# 2025-07-08

- [ ] implement VAPOR in run.py formalism
- [ ] clean up api.py to allow for choice of config
- [ ] get run.py to run on devbox & skypilot
- [ ]

storytelling meeting: we are wholly focused on navigation now. we don't have policies that don't wall-follow in open fields. Emmett is very insistent that we continue in this direction.

mat contrastive learning: https://github.com/Metta-AI/metta/tree/msb_CL_v2. contrastive learning, because of its dependence on model hidden states, is kind of a self-regularization in the loss. we're running into performance issues (10sps vs 65sps), so we probably have to write some GPU kernels to make it run faster.

curriculum learning is just a matter of using a curriculum config.

i kind of want to solve navigation :P. i think we can solve this by using a transformer architecture, basic loss, and better hyperparam sweeps.

basics:
- [ ] run contrastive RL 


# 2025-07-07

- Compositionality
	- this seems to be a harder research direction to study properly. it is the sort of thing i'm interested in long-term.
- Competence
- General SWE / QoL
	- understand how sweeps work (has the PR been approved?)
	- get a working implementation of VAPOR (more for pride than anything else)
	- what is the sort of exploration loss that George / Daphne talked about? how does it differ from VAPOR?
	- what is curriculum learning? what is the contrastive learning approach Matt is working on?
	- where does the generalization from training on basic-easy break?
	- what are the different environments available in mettagrid? what are the various configs you can make? (should I make anki cards for this?)
	- what are the other researchers doing here?
	- regularization?
	-
- other ~~major~~ interesting open research hypotheses (not biasing towards generality as much as I would like)
	- H4.2: "There is a critical convexity (steepness) threshold of reward shaping below which learning fails."
	- H9: "Agents trained on many sequences can generalize the abstract skill of sequence completion. Partially confirmed, depends on reward structure."
	- H30: "Careful optimisation of PPO/RL hyper-parameters (learning-rate, clip-ratio, entropy-coef, batch-size) is sufficient to solve currently failing associative-memory tasks without architectural changes. Untested — requires systematic sweeps."
	- H8.4.1: "It is unknown whether explicitly adding an exploration loss improves performance, what kind of exploration loss (e.g. entropy, curiosity, visitation novelty), and under what specific task conditions it is effective. Unknown — research needed to isolate and test under controlled regimes."
	- H31: "Training steps required grow faster than linearly with sequence length (or reward sparsity); current runs are orders-of-magnitude too short for long sequences. Preliminary evidence from simple graph experiments."
	- H32: "Narrower action spaces accelerate learning; expanding the action set dramatically slows convergence. Anecdotal evidence — needs controlled ablation."

which of these should I do before lunch? (1hr). i'm biasing towards learning about sweeps. (axel's PR hasn't been merged yet, he comes back on Monday). will ask o3 to explain the framework in various ways, otherwise this isn't a reasonable thing to work on. H4.2 is an interesting thing to think about. i guess this basically has to do with the epsilon rewards in reward sequences in the environment. i want to be careful about being robust to HPs though. (is there a way I can test how robust behavior is to HPs? maybe take an existing good policy and then run a sweep on it?). maybe i should think about how to best integrate exploration loss?

how is associative memory failing? is this basically lack of generalization across curricula?

re: adding new objects --- want to look at the environment and look at the typologies of game theoretic scenarios that result from different environment configurations in the environment. RL can't learn the optimal policy because sometimes the optimal policy in n-player games is nondeterministic (but in practice models can probably model internally policies that implement some pseudorandom thing?)

ok i should use pufferlib for hyper sweeps. (pufferlib for hyper sweeps did not work, or at least was not intuitive. feel like i have a marginally better understanding of how our configs work, or at least how hydra works, and i'm waiting on axel's new merge today for our in-house sweep implementation)


