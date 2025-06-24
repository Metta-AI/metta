# Metta AI

<p align="center">
  <a href="https://deepwiki.com/Metta-AI/metta">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
</p>

A reinforcement learning codebase focusing on the emergence of cooperation and alignment in multi-agent AI systems.

- **Discord**: <https://discord.gg/mQzrgwqmwy>
- **Short (5m) Talk**: <https://www.youtube.com/watch?v=bt6hV73VA8I>
- **Talk**: <https://foresight.org/summary/david-bloomin-metta-learning-love-is-all-you-need/>

## What is Metta Learning?

<p align="middle">
<img src="docs/readme_showoff.gif" alt="Metta learning example video">
<br>
<a href="https://metta-ai.github.io/metta/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fandre_pufferbox_33%2Freplay.77200.json.z&play=true">Interactive demo</a>
</p>

Metta AI is an open-source research project investigating the emergence of cooperation and alignment in multi-agent AI systems. By creating a model organism for complex multi-agent gridworld environments, the project aims to study the impact of social dynamics, such as kinship and mate selection, on learning and cooperative behaviors of AI agents.

Metta AI explores the hypothesis that social dynamics, akin to love in biological systems, play a crucial role in the development of cooperative AGI and AI alignment. The project introduces a novel reward-sharing mechanism mimicking familial bonds and mate selection, allowing researchers to observe the evolution of complex social behaviors and cooperation among AI agents. By investigating this concept in a controlled multi-agent setting, the project seeks to contribute to the broader discussion on the path towards safe and beneficial AGI.

## Introduction

Metta is a simulation environment (game) designed to train AI agents capable of meta-learning general intelligence. The core idea is to create an environment where incremental intelligence is rewarded, fostering the development of generally intelligent agents.

### Motivation and Approach

1. **Agents and Environment**: Agents are shaped by their environment, learning policies that enhance their fitness. To develop general intelligence, agents need an environment where increasing intelligence is continually rewarded.

2. **Competitive and Cooperative Dynamics**: A game with multiple agents and some competition creates an evolving environment where challenges increase with agent intelligence. Purely competitive games often reach a Nash equilibrium, where locally optimal strategies are hard to deviate from. Adding cooperative dynamics introduces more behavioral possibilities and smooths the behavioral space.

3. **Kinship Structures**: The game features a flexible kinship structure, simulating a range of relationships from close kin to strangers. Agents must learn to coordinate with close kin, negotiate with more distant kin, and compete with strangers. This diverse social environment encourages continuous learning and intelligence growth.

The game is designed to evolve with the agents, providing unlimited learning opportunities despite simple rules.

### Game Overview

The current version of the game can be found [here](https://huggingface.co/metta-ai/baseline.v0.1.0). It's a grid world with the following dynamics:

- **Agents and Vision**: Agents can see a limited number of squares around them.
- **Resources**: Agents harvest diamonds, convert them to energy at charger stations, and use energy to power the "heart altar" for rewards.
- **Energy Management**: All actions cost energy, so agents learn to manage their energy budgets efficiently.
- **Combat**: Agents can attack others, temporarily freezing the target and stealing resources.
- **Defense**: Agents can toggle shields, which drain energy but absorb attacks.
- **Cooperation**: Agents can share energy or resources and use markers to communicate.

### Exploration and Expansion

The game offers numerous possibilities for exploration, including:

1. **Diverse Energy Profiles**: Assigning different energy profiles to agents, essentially giving them different bodies and policies.
2. **Dynamic Energy Profiles**: Allowing agents to change their energy profiles, reflecting different postures or emotions.
3. **Resource Types and Conversions**: Introducing different resource types and conversion mechanisms.
4. **Environment Modification**: Enabling agents to modify the game board by creating, destroying, or altering objects.

### Kinship and Social Dynamics

The game explores various kinship structures:

1. **Random Kinship Scores**: Each pair of agents has a kinship score sampled from a distribution.
2. **Teams**: Agents belong to teams with symmetric kinship among team members.
3. **Hives/Clans/Families**: Structuring agents into larger kinship groups.

Future plans include incorporating mate-selection dynamics, where agents share future rewards at a cost, potentially leading to intelligence gains through a signaling arms race.

Metta aims to create a rich, evolving environment where AI agents can develop general intelligence through continuous learning and adaptation.

## Research Explorations

The project's modular design and open-source nature make it easy for researchers to adapt and extend the platform to investigate their own hypotheses in this domain. The highly performant, open-ended game rules provide a rich environment for studying these behaviors and their potential implications for AI alignment.

Some areas of research interest:

#### 1. Environment Development

Develop rich and diverse gridworld environments with complex dynamics, such as resource systems, agent diversity, procedural terrain generation, support for various environment types, population dynamics, and kinship schemes.

#### 2. Agent Architecture Research

Incorporate techniques like dense learning signals, surprise minimization, exploration strategies, and blending reinforcement and imitation learning.

#### 3. Scalable Training Infrastructure

Investigate scalable training approaches, including distributed reinforcement learning, student-teacher architectures, and blending reinforcement learning with imitation learning, to enable efficient training of large-scale multi-agent systems.

#### 4. Intelligence Evaluations for Gridworld Agents

Design and implement a comprehensive suite of intelligence evaluations for gridworld agents, covering navigation tasks, maze solving, in-context learning, cooperation, and competition scenarios.

#### 5. DevOps and Tooling

Develop tools and infrastructure for efficient management, tracking, and deployment of experiments, such as cloud cluster management, experiment tracking and visualization, and continuous integration and deployment pipelines.

This README provides only a brief overview of research explorations. Visit the [research roadmap](https://github.com/Metta-AI/metta/blob/main/roadmap.md) for more details.

## Installation

Install uv (a fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional: run the script which will configure the development environment (the script might fail if you're not on the Metta dev team and don't have permissions):

```bash
./devops/setup_dev.sh
```

After git updates, you might need to run `uv sync` to reinstall all necessary dependencies.

## Training a Model

### Run the training

```
./tools/train.py run=my_experiment +hardware=macbook wandb=off
```

`run` names your experiment and controls where checkpoints are saved under
`train_dir/<run>`. Hardware presets such as `+hardware=macbook` tune the trainer
for your machine. You can pass `+user=<name>` to load defaults from
`configs/user/<name>.yaml`. Use `wandb=off` to disable Weights & Biases logging
if you don't have access.

## Visualizing a Model

### Run the interactive simulation

```
./tools/play.py run=my_experiment +hardware=macbook wandb=off
```

This launches a human-controlled session using the same configuration flags as
training. It is useful for quickly testing maps or policies on your local
hardware.

### Run the terminal simulation

```
./tools/renderer.py run=demo_obstacles \
renderer_job.environment.uri="configs/env/mettagrid/maps/debug/simple_obstacles.map"
```

## Evaluating a Model

When you run training, if you have WandB enabled, then you will be able to see in your WandB run page results for the eval suites.

However, this will not apply for anything trained before April 8th.

### Post Hoc Evaluation

If you want to run evaluation post-training to compare different policies, you can do the following:

To add your policy to the existing navigation evals DB:

```
./tools/sim.py \
    sim=navigation \
    run=navigation101 \
    policy_uri=wandb://run/YOUR_POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navigation_db \
    device=cpu
```

This will run your policy through the `configs/eval/navigation` eval_suite and then save it to the `navigation_db` artifact on WandB.

Then, to see the results in the heatmap along with the other policies in the database, you can run:

```
./tools/dashboard.py +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html
```

## Development Setup

To run the style checks and tests locally:

```bash
ruff format
ruff check
pyright metta  # optional, some stubs are missing
pytest
```

Running these commands mirrors our CI configuration and helps keep the codebase
consistent.

## Third-party Content

Some sample map patterns in `scenes/dcss` were adapted from the open-source game [Dungeon Crawl Stone Soup (DCSS)](https://github.com/crawl/crawl),
specifically from the file [`simple.des`](https://github.com/crawl/crawl/blob/master/crawl-ref/source/dat/des/arrival/simple.des).

DCSS is licensed under the [GNU General Public License v2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).
