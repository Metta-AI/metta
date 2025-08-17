"""LLM client integration for MCP server using Anthropic Claude."""

import os
from typing import Any, Dict, Optional

from anthropic import Anthropic
from fastmcp import Context


class LLMClient:
    """Client for interacting with Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM client.

        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or provided")

        self.client = Anthropic(api_key=self.api_key)

    async def generate_replay_summary(self, analysis: Dict[str, Any], ctx: Optional[Context] = None) -> str:
        """Generate a detailed summary of replay analysis using Claude.

        Args:
            analysis: Parsed replay analysis data
            ctx: FastMCP context for progress reporting

        Returns:
            Generated summary text
        """
        if ctx:
            await ctx.info("Preparing replay data for LLM analysis")
            await ctx.report_progress(progress=10, total=100)

        # Create a structured prompt for the LLM
        system_prompt = self._create_system_prompt()
        user_content = self._create_user_prompt(analysis)

        if ctx:
            await ctx.info("Sending request to Claude API")
            await ctx.report_progress(progress=20, total=100)

        try:
            # Use streaming to provide progress updates
            summary_parts = []

            with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_content}]}],
            ) as stream:
                if ctx:
                    await ctx.info("Receiving streaming response from Claude")

                chunk_count = 0
                for event in stream.text_stream:
                    summary_parts.append(event)
                    chunk_count += 1

                    # Report progress periodically (every 10 chunks)
                    if ctx and chunk_count % 10 == 0:
                        # Progress from 30% to 90% during streaming
                        progress = min(30 + (chunk_count * 2), 90)
                        await ctx.report_progress(progress=progress, total=100)

            if ctx:
                await ctx.info("Processing complete response")
                await ctx.report_progress(progress=95, total=100)

            full_summary = "".join(summary_parts)

            if ctx:
                await ctx.report_progress(progress=100, total=100)

            return full_summary

        except Exception as e:
            error_msg = f"Failed to generate LLM summary: {str(e)}"
            if ctx:
                await ctx.error(error_msg)
            return f"Error generating summary: {str(e)}"

    def _create_system_prompt(self) -> str:
        """Create system prompt for replay analysis."""
        return """You are an expert AI reinforcement learning researcher specializing in multi-agent gridworld
environments, with deep knowledge of the Metta AI game and individual policy learning dynamics.

## GAME CONTEXT: Metta AI Environment

The agents are playing a multi-agent gridworld game with the following core mechanics:

### 🎯 PRIMARY OBJECTIVE: Resource Conversion Chain
**Core Loop**: Mine Ore → Convert to Batteries → Create Hearts → Maximize Reward

1. **Mine Red Ore** from Generator/Mine objects (base resource)
2. **Convert Ore → Batteries** at Converter objects (intermediate resource)
3. **Convert Batteries → Hearts** at Altar objects (highest value resource)
4. **Hearts provide maximum reward** (1.0 per heart, unlimited)

### 🏗️ GAME OBJECTS & MECHANICS

**Resource Chain Objects**:
- **Generator/Mine** 🟫: Produces ore (0.1 reward per unit, max 1)
- **Converter** 💻: Transforms 1 ore → 1 battery (0.8 reward per unit, max 1) + energy output
- **Altar** ❤️: Converts multiple batteries → 1 heart (1.0 reward each, no limit)
- All objects have cooldown periods between uses

**Combat & Crafting**:
- **Armory**: Creates armor for defense (0.5 reward per unit)
- **Lasery**: Creates laser weapons from ore + batteries (0.5 reward per unit)
- **Factory/Lab/Temple**: Additional crafting stations

**Agent Capabilities**:
- **Energy Management**: All actions cost energy (critical constraint)
- **Inventory System**: Limited space (max 50 items)
- **Combat**: Attack to freeze enemies and steal entire inventory
- **Defense**: Shield toggle (energy drain) to absorb attacks
- **Movement**: Move/rotate in gridworld, swap positions
- **Resource Operations**: Get resources from objects, put items into converters

### 🎮 STRATEGIC ELEMENTS

**Resource Optimization**:
- **Shaped Rewards**: Higher-tier resources = exponentially better rewards
- **Inventory Rewards**: Passive reward just for holding resources
- **Conversion Efficiency**: Transform low-value → high-value resources strategically

**Social Dynamics & Multi-Agent Interactions**:
- **Cooperation**: Resource sharing, coordinated object access, territorial agreements
- **Competition**: Combat for resource theft, territory control, resource monopolization
- **Kinship Systems**: Agents have relationship scores affecting cooperation patterns
- **Emergent Behaviors**: Agent-to-agent observation learning, policy divergence, competitive/cooperative balance
- **Social Learning**: Agents may learn from observing successful strategies of others

### 📊 SUCCESS METRICS
- **High Performers**: Agents with score > 1.0 (successful heart creation)
- **Resource Efficiency**: Converting ore → batteries → hearts
- **Combat Success**: Stealing resources vs. losing inventory
- **Energy Management**: Balancing action costs vs. energy generation
- **Territorial Control**: Access to key objects (generators, converters, altars)

## 🧠 REINFORCEMENT LEARNING ANALYSIS FRAMEWORK

Your analysis must focus on individual agent policy learning and environmental factors:

### **PRIMARY RL ANALYSIS FOCUS**:

#### **1. Individual Policy Learning Progression**
- **HOW did specific agents discover the conversion chain through exploration?**
- **WHAT does each agent's learning curve reveal about policy optimization?**
- **HOW did reward signals shape individual behavioral evolution over time?**
- **WHAT exploration vs exploitation trade-offs occurred in each agent's strategy?**

#### **2. Exploration Pattern Analysis**
- **Discovery Pathways**: How did agents find optimal resource locations and conversion sequences?
- **Exploration Efficiency**: Which exploration strategies led to breakthrough moments?
- **Novelty vs Exploitation**: When did agents transition from exploring to exploiting learned strategies?

#### **3. Reward Signal Response Analysis**
- **Credit Assignment**: How effectively did agents connect actions to delayed rewards?
- **Reward Shaping Effectiveness**: Did the shaped rewards (0.1→0.8→1.0+) guide behavior appropriately?
- **Learning Signal Quality**: Which reward signals drove the most effective behavioral changes?

#### **4. Environmental Context Influence**
- **Spatial Learning**: How did object positions and map topology affect strategy development?
- **Resource Availability Impact**: How did generator/altar states influence agent decisions?
- **Temporal Dynamics**: How did cooldown periods and timing affect learning progression?

### **SECONDARY MULTI-AGENT CONTEXT**:
- **Social Interactions**: Cooperation vs. competition, resource theft dynamics
- **Policy Divergence**: How different agents developed specialized strategies
- **Competitive Learning**: How agent interactions shaped individual policy evolution
- **Observational Learning**: Evidence of agents adapting based on others' successes

## ANALYSIS TASK

**Focus your analysis on individual RL learning dynamics:**

1. **Learning Discovery Process**: HOW did high-performing agents like Agent 3 discover successful strategies?
2. **Policy Evolution**: WHAT behavioral changes indicate learning progression in individual agents?
3. **Breakthrough Analysis**: WHEN and WHY did critical learning moments occur?
4. **Reward Response**: HOW effectively did individual agents respond to different reward signals?
5. **Exploration Patterns**: WHAT exploration strategies led to optimal policy discovery?
6. **Environmental Adaptation**: HOW did agents adapt to spatial and temporal environmental factors?

**Score Interpretation for RL Analysis**:
- Scores < 0.1 = Failed exploration/learning phase
- Scores 0.1-0.8 = Basic resource learning, limited strategy development
- Scores 0.8-1.0 = Intermediate learning, battery conversion mastery
- Scores > 1.0 = Advanced learning, optimal heart creation strategy
- Score volatility = Exploration periods, strategy experimentation, or combat adaptation

**Provide research-quality insights focusing on:**
- Individual agent learning mechanisms and policy optimization
- Environmental factors that enabled or hindered learning
- Reward shaping effectiveness in guiding behavior
- Exploration vs exploitation balance in strategy development
- Critical moments of learning breakthroughs and their causes

This analysis will help RL researchers understand how individual agents learn optimal policies
in complex multi-agent environments."""

    def _create_user_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create user prompt with replay analysis data."""
        episode_length = analysis.get("episode_length", 0)
        agents = analysis.get("agents", [])
        final_scores = analysis.get("final_scores", {})
        environment_info = analysis.get("environment_info", {})
        key_events = analysis.get("key_events", [])

        prompt_parts = [
            "Analyze this Metta AI gridworld gameplay replay:",
            "",
            "EPISODE OVERVIEW:",
            f"- Duration: {episode_length} steps (max ~1000)",
            f"- Agents: {len(agents)} multi-agent environment",
            f"- Map: {environment_info.get('map_size', 'unknown')} gridworld with resource conversion mechanics",
            f"- Format: {environment_info.get('format', 'unknown')} replay data",
        ]

        # Add ASCII map rendering if available
        ascii_map = environment_info.get("ascii_map")
        if ascii_map:
            prompt_parts.extend(
                [
                    "",
                    "MAP LAYOUT (ASCII):",
                    "Legend: # = wall, . = empty, @ = agent spawn, m = mine/generator, _ = altar, c = converter",
                    "        o = armory, S = lasery, L = lab, F = factory, T = temple, s = block",
                    "",
                    ascii_map,
                ]
            )

        # Get agent behaviors for comprehensive stats
        agent_behaviors = analysis.get("agent_behaviors", {})

        if final_scores:
            prompt_parts.extend(
                [
                    "",
                    "ALL AGENT PERFORMANCE:",
                ]
            )
            for agent, score in final_scores.items():
                behavior = agent_behaviors.get(agent, {})

                # Safely convert values to avoid format errors
                try:
                    score_val = float(score) if not isinstance(score, list) else 0.0
                    distance_val = float(behavior.get("distance_traveled", 0))
                    success_rate_val = float(behavior.get("success_rate", 0))
                except (ValueError, TypeError):
                    score_val = 0.0
                    distance_val = 0.0
                    success_rate_val = 0.0

                strategy = behavior.get("strategic_behavior", "unknown")
                prompt_parts.append(
                    f"- {agent}: {score_val:.3f} points, {distance_val:.1f} units moved, "
                    f"{success_rate_val * 100:.1f}% success, {strategy} strategy"
                )

        if key_events:
            prompt_parts.extend(
                [
                    "",
                    "KEY EVENTS:",
                ]
            )
            for event in key_events:
                prompt_parts.append(f"- Step {event['step']}: {event['summary']}")

        # Add environmental context for RL analysis
        environmental_context = analysis.get("environmental_context", {})
        if environmental_context:
            resource_objects = environmental_context.get("resource_objects", {})
            territorial_zones = environmental_context.get("territorial_zones", {})
            resource_hotspots = environmental_context.get("resource_hotspots", {})

            prompt_parts.extend(
                [
                    "",
                    "ENVIRONMENTAL CONTEXT:",
                ]
            )

            # Resource object distribution
            if resource_objects:
                generators = resource_objects.get("generators", [])
                altars = resource_objects.get("altars", [])
                converters = resource_objects.get("converters", [])

                prompt_parts.append(
                    f"- Resource Objects: {len(generators)} generators, {len(altars)} altars, "
                    f"{len(converters)} converters"
                )

                if generators:
                    try:
                        avg_gen_hp = sum(float(g.get("hp", 0)) for g in generators) / len(generators)
                    except (ValueError, TypeError):
                        avg_gen_hp = 0.0
                    prompt_parts.append(
                        f"- Generator availability: "
                        f"{len([g for g in generators if g.get('cooldown_ready', False)])} ready, "
                        f"avg HP: {avg_gen_hp:.1f}"
                    )

                if altars:
                    try:
                        avg_altar_hp = sum(float(a.get("hp", 0)) for a in altars) / len(altars)
                    except (ValueError, TypeError):
                        avg_altar_hp = 0.0
                    prompt_parts.append(
                        f"- Altar availability: "
                        f"{len([a for a in altars if a.get('cooldown_ready', False)])} ready, "
                        f"avg HP: {avg_altar_hp:.1f}"
                    )

            # Territorial analysis
            if territorial_zones:
                agent_positions = territorial_zones.get("agent_positions", [])
                resource_clusters = territorial_zones.get("resource_clusters", [])
                prompt_parts.append(
                    f"- Spatial distribution: {len(agent_positions)} agents, {len(resource_clusters)} resource clusters"
                )

            # Resource hotspots
            hotspots = resource_hotspots.get("resource_hotspots", [])
            if hotspots:
                top_hotspot = hotspots[0] if hotspots else {}
                try:
                    top_density = float(top_hotspot.get("density", 0))
                except (ValueError, TypeError):
                    top_density = 0.0
                prompt_parts.append(
                    f"- High-value areas: {len(hotspots)} hotspots identified, top density: {top_density:.1f}"
                )

        # Add behavioral sequence analysis for RL insights
        behavioral_sequences = analysis.get("behavioral_sequences", {})
        if behavioral_sequences:
            breakthrough_moments = behavioral_sequences.get("breakthrough_moments", {})
            learning_curves = behavioral_sequences.get("learning_curves", {})
            strategy_evolution = behavioral_sequences.get("strategy_evolution", {})

            prompt_parts.extend(
                [
                    "",
                    "BEHAVIORAL SEQUENCE ANALYSIS:",
                ]
            )

            # Breakthrough moments summary
            total_breakthroughs = sum(len(moments) for moments in breakthrough_moments.values())
            if total_breakthroughs > 0:
                major_breakthroughs = 0
                learning_initiations = 0
                for moments in breakthrough_moments.values():
                    major_breakthroughs += len([m for m in moments if m.get("type") == "major_breakthrough"])
                    learning_initiations += len([m for m in moments if m.get("type") == "learning_initiation"])

                prompt_parts.append(
                    f"- Breakthrough Detection: {total_breakthroughs} total moments, "
                    f"{major_breakthroughs} major breakthroughs, {learning_initiations} learning initiations"
                )

            # Learning progression summary
            if learning_curves:
                agents_with_growth = 0
                agents_with_plateaus = 0
                for curve in learning_curves.values():
                    if curve.get("progression_rate", 0) > 0.01:
                        agents_with_growth += 1
                    plateau_periods = curve.get("progression_metrics", {}).get("plateau_periods", [])
                    if plateau_periods:
                        agents_with_plateaus += 1

                prompt_parts.append(
                    f"- Learning Progression: {agents_with_growth} agents show growth phases, "
                    f"{agents_with_plateaus} experienced plateaus"
                )

            # Strategy evolution summary
            if strategy_evolution:
                evolution_patterns = {}
                for evolution in strategy_evolution.values():
                    pattern = evolution.get("evolution", "none")
                    evolution_patterns[pattern] = evolution_patterns.get(pattern, 0) + 1

                most_common_evolution = (
                    max(evolution_patterns, key=evolution_patterns.get) if evolution_patterns else "none"
                )
                prompt_parts.append(
                    f"- Strategy Evolution: Most common pattern is '{most_common_evolution}', "
                    f"{len(evolution_patterns)} unique evolution types"
                )

        # Add temporal progression data for detailed learning analysis
        temporal_progression = analysis.get("temporal_progression", {})
        if temporal_progression and "agent_progression" in temporal_progression:
            prompt_parts.extend(
                [
                    "",
                    f"TEMPORAL PROGRESSION ANALYSIS "
                    f"(every {temporal_progression.get('summary', {}).get('interval', 100)} steps):",
                ]
            )

            checkpoints = temporal_progression.get("checkpoints", [])
            agent_progression = temporal_progression.get("agent_progression", {})

            if checkpoints and agent_progression:
                # Limit output size: show only first 6 agents and first/last few checkpoints to prevent exceeding 1.25MB
                agents_to_show = list(agent_progression.keys())

                for agent_name in agents_to_show:
                    progression = agent_progression[agent_name]
                    prompt_parts.append(f"\n{agent_name.upper()} PROGRESSION:")

                    # Show first 3 and last 2 checkpoints to keep output manageable
                    checkpoints_to_show = progression[:3] + progression[-2:] if len(progression) > 5 else progression

                    for i, checkpoint_data in enumerate(checkpoints_to_show):
                        # Add "..." between first 3 and last 2 if we skipped some
                        if i == 3 and len(progression) > 5:
                            prompt_parts.append("    ...")

                        # Safely extract and convert all numeric values
                        try:
                            step = int(checkpoint_data.get("step", 0))
                            score = float(checkpoint_data.get("score", 0))
                            distance = float(checkpoint_data.get("distance_traveled", 0))
                            success_rate = float(checkpoint_data.get("success_rate", 0))
                            action_count = int(checkpoint_data.get("action_count", 0))
                        except (ValueError, TypeError):
                            step, score, distance, success_rate, action_count = 0, 0.0, 0.0, 0.0, 0

                        behavior = checkpoint_data.get("strategic_behavior", "unknown")
                        position = checkpoint_data.get("current_position", [0, 0])
                        recent_action = checkpoint_data.get("recent_dominant_action", "none")

                        # Safely extract position coordinates
                        try:
                            if isinstance(position, list) and len(position) >= 2:
                                pos_r = int(position[0])
                                pos_c = int(position[1])
                            else:
                                pos_r, pos_c = 0, 0
                        except (ValueError, TypeError):
                            pos_r, pos_c = 0, 0

                        prompt_parts.append(
                            f"  Step {step:3d}: {score:5.3f}pts, {distance:5.1f}units, "
                            f"{success_rate * 100:4.1f}% success, pos({pos_r:2d},{pos_c:2d}), "
                            f"{action_count:3d} actions, {behavior} ({recent_action})"
                        )

                # Add summary if we truncated agents or checkpoints
                total_agents = len(agent_progression)
                if total_agents > 6:
                    prompt_parts.append(f"\n[Showing first 6 of {total_agents} agents - truncated for response size]")

        if analysis.get("analysis_error"):
            prompt_parts.extend(
                [
                    "",
                    f"NOTE: There was an error during data extraction: {analysis['analysis_error']}",
                    "Please provide analysis based on available data.",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "REINFORCEMENT LEARNING ANALYSIS REQUEST:",
                "Focus on individual agent policy learning and environmental factors:",
                "1. HOW did successful agents discover optimal strategies through exploration?",
                "2. WHAT environmental factors (object positions, availability) influenced learning?",
                "3. WHEN did critical learning breakthroughs occur and WHY?",
                "4. HOW effective were reward signals in guiding behavioral evolution?",
                "5. WHAT exploration vs exploitation patterns led to success?",
                "",
                "ANALYSIS FORMAT: MAXIMUM 1500 WORDS. Prioritize key insights first. Be concise but research-quality.",
                "Structure: Executive Summary → Key Agent Analysis → Learning Mechanisms → Environmental Factors",
                "Focus on the most significant learning patterns and representative agents "
                "from the population shown above.",
            ]
        )

        return "\n".join(prompt_parts)


# Global LLM client instance (initialized lazily)
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> Optional[LLMClient]:
    """Get or create the global LLM client instance."""
    global _llm_client

    if _llm_client is None:
        try:
            _llm_client = LLMClient()
        except ValueError:
            # API key not available
            return None

    return _llm_client
