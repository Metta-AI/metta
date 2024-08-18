# Metta AI
A reinforcement learning codebase focusing on the emergence of cooperation and alignment in multi-agent AI systems.

* **Discord**: https://discord.gg/mQzrgwqmwy
* **Talk**: https://foresight.org/summary/david-bloomin-metta-learning-love-is-all-you-need/

## What is Metta Learning?
<p align="middle">
<img src="https://github.com/debbly/metta-learning-assets/blob/main/gifs/example_video.gif?raw=true" width="360" alt="Metta learning example video">
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

This readme provides only a brief overview of research explorations. Visit the [research roadmap](https://github.com/daveey/metta/blob/master/roadmap.md) for more details.

## Installation

### Install the Vulkan SDK

<details>
<summary>Mac</summary>

[Download Vulkan SDK for Mac](https://sdk.lunarg.com/sdk/download/1.3.224.1/mac/vulkansdk-macos-1.3.224.1.dmg)

</details>

<details>
<summary>Windows</summary>

[Download Vulkan SDK for Windows](https://sdk.lunarg.com/sdk/download/1.3.224.1/windows/VulkanSDK-1.3.224.1-Installer.exe)

</details>

<details>
<summary>Linux</summary>

```bash
mkdir ~/vulkan
cd ~/vulkan
wget https://sdk.lunarg.com/sdk/download/1.3.224.1/linux/vulkansdk-linux-x86_64-1.3.224.1.tar.gz
tar -zxvf vulkansdk-linux-x86_64-1.3.224.1.tar.gz
cd 1.3.224.1
yes | ./vulkansdk
echo 'source ~/1.3.224.1/setup-env.sh' >> ~/.bashrc
```
</details>

### Install the requirements
Metta uses griddly and sample factory repos as submodules. To fetch the latest versions of these projects use the following commands:
```
git submodule init
git submodule update
```
Start installing the requiruments:
```
conda create -n metta python=3.11.7
conda activate metta
pip install -e .
```

### (Optional) Setup HuggingFace Access

1. Create a HuggingFace account
2. Create a token in your account settings
3. Run `huggingface-cli login` and paste the token

# Running a Simulation

### Download a baseline model
To download the model files, install git-lfs first, then run the following commands:

```
git lfs install
./devops/load_model.sh baseline.v0.5.4
```

### Render a video of the evaluation

```
python -m tools.run cmd=evaluate experiment=baseline.v0.5.4
```

# Training a Model

### Run the training

```
python -m tools.run cmd=train experiment=my_experiment hardware=macbook
```


### Troubleshooting
#### conan installation
In case of an an "Invalid setting" error when running `pip install`, e.g., :
```
"ERROR: Invalid setting '15' is not a valid 'settings.compiler.version' value.
Possible values are ['5.0', ..., '14.0']
Read "http://docs.conan.io/en/latest/faq/troubleshooting.
html#error-invalid-setting"
```
Follow these steps (tested on MacOS):
1. open ~/.conan/settings.yml
2. Inside the file, go to `apple-clang` -> `version` and add missing value ("15" in the example above) is in the `version` list
3. For more information: https://docs.conan.io/en/1.16/faq/troubleshooting.html
