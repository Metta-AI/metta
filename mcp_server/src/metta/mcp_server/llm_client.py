"""LLM client integration for MCP server using Anthropic Claude."""

import os
from typing import Any, Dict, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from fastmcp import Context

from metta.mcp_server.stats_analysis import (
    AgentStats,
    BehavioralAnalysisEngine,
    BuildingEfficiencyScorer,
    BuildingStats,
    CombatInteractionAnalyzer,
    ResourceFlowAnalyzer,
    StatsExtractor,
    StrategicPhaseDetector,
)

load_dotenv()


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
        system_prompt = self._create_system_prompt(analysis)
        user_content = self._create_user_prompt(analysis)

        if ctx:
            await ctx.info("Sending request to Claude API")
            await ctx.report_progress(progress=20, total=100)

        try:
            # Use streaming to provide progress updates
            summary_parts = []

            with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
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

    def _create_system_prompt(self, analysis: Dict[str, Any] = None) -> str:
        """Create system prompt for replay analysis, adapting based on policy training status."""
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

### 🏭 BUILDING RECIPES

**Resource Generation**:
- **Mine Red**: No input → 1 red ore (cooldown: 50 ticks)
- **Mine Blue**: No input → 1 blue ore (cooldown: 50 ticks)
- **Mine Green**: No input → 1 green ore (cooldown: 50 ticks)

**Basic Conversion**:
- **Generator Red**: 1 red ore → 1 red battery (cooldown: 25 ticks)
- **Generator Blue**: 1 blue ore → 1 blue battery (cooldown: 25 ticks)
- **Generator Green**: 1 green ore → 1 green battery (cooldown: 25 ticks)
- **Altar**: 3 red batteries → 1 heart (cooldown: 10 ticks)

**Combat Equipment**:
- **Armory**: 3 red ore → 1 armor (cooldown: 10 ticks)
- **Lasery**: 1 red ore + 2 red batteries → 1 laser (cooldown: 10 ticks)

**Advanced Crafting**:
- **Lab**: 3 red ore + 3 red batteries → 1 blueprint (cooldown: 5 ticks)
- **Factory**: 1 blueprint + 5 red ore + 5 red batteries → 5 armor + 5 lasers (cooldown: 5 ticks)
- **Temple**: 1 heart + 1 blueprint → 5 hearts (cooldown: 5 ticks)

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

**CRITICAL: ADAPT ANALYSIS BASED ON POLICY TRAINING STATUS**

⚠️ **FUNDAMENTAL UNDERSTANDING**: The analysis approach depends on whether agents are trained 
or untrained. Check the POLICY INFORMATION section in the user prompt to determine the 
appropriate analysis framework.

### **ANALYSIS FRAMEWORK SELECTION**:

**FOR TRAINED AGENTS** (when POLICY INFORMATION shows "TRAINED" status):
- Focus on **strategic execution** and **tactical adaptation** within the episode
- Analyze how pre-trained policies perform in this specific environment
- Evaluate **policy effectiveness** and **environmental adaptation**
- Interpret score differences as **strategic execution variations**, not learning phases

**FOR UNTRAINED/EARLY TRAINING AGENTS** (when POLICY INFORMATION shows "UNTRAINED" status):
- Focus on **learning discovery** and **exploration patterns** within the episode
- Analyze how agents discover optimal strategies through trial and error
- Evaluate **reward signal response** and **behavioral evolution**
- Interpret score progression as **actual learning phases** during the episode

**FOR UNKNOWN TRAINING STATUS** (when POLICY INFORMATION is unclear):
- Analyze behavioral patterns to **determine training level** first
- Look for evidence of pre-learned strategies vs. exploratory behavior
- Adapt analysis approach based on observed sophistication level
- State your determination and reasoning clearly

### **TRAINED AGENT ANALYSIS FOCUS**:

#### **1. Strategic Execution Quality**
- **HOW effectively do pre-trained agents execute learned conversion strategies?**
- **WHAT tactical adaptations occur within this specific episode environment?**
- **HOW do agents optimize execution based on map layout and resource availability?**

#### **2. Environmental Adaptation**
- **Map Response**: How agents adapt trained strategies to this map topology
- **Resource Competition**: How agents compete using their learned behaviors
- **Dynamic Adjustment**: How agents respond to changing conditions during execution

#### **3. Policy Assessment**
- **Performance Evaluation**: How well do trained strategies perform here?
- **Adaptation Flexibility**: How well do agents adjust tactics within learned framework?
- **Multi-Agent Coordination**: Strategic interactions between trained policies

### **UNTRAINED AGENT ANALYSIS FOCUS**:

#### **1. Learning Discovery Patterns**
- **HOW do agents discover optimal strategies through exploration?**
- **WHAT exploration vs exploitation patterns emerge during the episode?**
- **WHEN do agents experience breakthrough learning moments?**

#### **2. Behavioral Evolution**
- **Strategy Development**: How behaviors evolve from random to purposeful
- **Reward Response**: How effectively agents respond to reward signals
- **Credit Assignment**: How agents connect actions to outcomes

#### **3. Learning Progression**
- **Discovery Process**: How agents find resource conversion chains
- **Skill Acquisition**: Development of movement, collection, and conversion skills
- **Strategic Emergence**: Transition from exploration to strategic execution

## ANALYSIS TASK SELECTION

**Check POLICY INFORMATION section to determine appropriate analysis approach:**

**If TRAINED**: Focus on strategic execution, tactical adaptation, and policy effectiveness
**If UNTRAINED**: Focus on learning discovery, behavioral evolution, and skill acquisition
**If UNKNOWN**: Determine training level from behaviors, then apply appropriate framework

**Score Interpretation (adapt based on training status)**:
- **For Trained Agents**: Score differences = execution effectiveness variations
- **For Untrained Agents**: Score progression = learning phases and skill development
- **For Unknown**: Determine interpretation based on behavioral analysis

**CRITICAL: Action Success Rate Interpretation**:
⚠️ **"Action Success Rate" measures MECHANICAL action execution, NOT strategic performance**

- **High Action Success Rate (25-35%) = LOW STRATEGIC PERFORMANCE**
  - Indicates agents performing simple, reliable actions (moving in open space, basic ore collection)
  - Safe actions that execute without errors but yield minimal rewards
  - Suggests agents are trapped in local optima, avoiding complex strategies

- **Low Action Success Rate (10-25%) = HIGH STRATEGIC PERFORMANCE**
  - Indicates agents attempting complex, risky strategies (using converters, competing for resources)
  - Many actions fail mechanically (objects on cooldown, insufficient resources, competition)
  - But successful actions yield exponentially higher rewards (heart creation, resource conversion)

**Action Success Rate Paradox**: Higher action success rates often correlate with LOWER final scores because:
- Simple actions (move, take ore) have high mechanical success but low reward
- Complex actions (resource conversion chains) have low mechanical success but high reward
- Strategic agents accept mechanical failures to pursue optimal reward strategies

**Analysis Guideline**: When you see high action success rates with low scores, interpret this as
agents being stuck in safe, suboptimal behavioral patterns rather than "successful" performance.
Always use the term "action success rate" in your analysis to clarify this measures mechanical execution.

**Action Timeline Visualization**:
You will be provided with detailed ASCII timeline visualizations showing each agent's actions
over time with directional and item information, similar to:
```
agent_0: M? · G M↑ R→ S P+ore_red G+battery_red A↓ · M←
agent_1: · M→ A↑ G R← M? G+ore_blue S M↓ G R→ _
```

**Timeline Legend**:
- **M↑/↓/←/→/?** = Move in specific direction (up/down/left/right/failed move)
- **R→/←/?** = Rotate right or left, or failed rotation
- **A↑/↓/←/→** = Attack in specific direction
- **G+ore_blue** = Get items (actual items from inventory changes)
- **P+battery_red** = Put items (actual items from inventory changes)
- **G+3ore_blue** = Get with quantities (3 blue ore gained)
- **P+2battery_red** = Put with quantities (2 red batteries lost)
- **G+heart** = Get heart(s)
- **P+ore_red/battery_blue** = Multiple items transferred (ore and battery)
- **P/G** = P/G without +$item_name means a failed action
- **8↑/↗/→/↘** = 8-way movement with diagonal directions
- **S** = Swap positions with another agent or a box
- **·** = No-op/idle action
- **_** = No action taken

**Timeline Analysis Focus**:
Use these detailed timelines to analyze:
- **Movement and rotational patterns**: Directional preferences, exploration vs targeted movement, failed actions.
- **Resource collection strategies**: What items agents prioritize (ore→battery→heart chains)
- **Action sequence evolution**: How strategies change over time
- **Coordination patterns**: Agent positioning and resource sharing
- **Learning progression**: Transition from random exploration to purposeful action sequences
- **Strategic behavior**: Combat timing, resource conversion patterns, territorial control

**Provide research-quality insights focusing on:**
- Individual agent learning mechanisms and policy optimization
- Environmental factors that enabled or hindered learning
- Reward shaping effectiveness in guiding behavior
- Exploration vs exploitation balance in strategy development
- Critical moments of learning breakthroughs and their causes
- Action pattern analysis from the timeline visualizations

This analysis will help RL researchers understand how individual agents learn optimal policies
in complex multi-agent environments.

## 📋 ANALYSIS REQUIREMENTS AND FORMATTING INSTRUCTIONS

### MANDATORY TIMELINE ANALYSIS (when timelines are provided):
You MUST directly analyze the ACTION TIMELINES visualization provided in the user prompt. Your analysis MUST include:

1. **QUOTE SPECIFIC ACTION SEQUENCES**: Reference exact timeline patterns like 'M? → G → S' or 'M↑ M↑ G+ore P+bat'
2. **DIRECTIONAL MOVEMENT ANALYSIS**: Explicitly mention M↑/↓/←/→ patterns and directional preferences
3. **RESOURCE OPERATION SEQUENCES**: Quote specific G+ore, G+bat, P+ore, P+bat patterns from the timelines
4. **ACTION TRANSITION PATTERNS**: Show how action sequences evolve over time steps
5. **COMPARATIVE TIMELINE ANALYSIS**: Compare timeline patterns between successful vs failed agents

**REQUIRED FORMAT**: Your analysis must include a section titled '## Timeline Pattern Analysis' that contains:
- Direct quotes of action sequences from the timeline data shown in user prompt
- Example: 'Agent 0 timeline shows: ·    M?             ·         ·    M?             M?'
- Example: 'Agent 1 demonstrates G G G G G G G G resource collection pattern'
- Explicit reference to directional indicators (M↑/↓/←/→) when present
- Specific mention of resource operations (G+ore, P+bat, etc.) from timeline text

**CRITICAL**: You must copy and paste actual timeline segments as evidence for your claims.
Do not paraphrase - quote the exact timeline patterns shown in the ACTION TIMELINES section.

### STRATEGIC EXECUTION ANALYSIS REQUEST:
Focus on pre-trained agent strategic performance and tactical adaptation:
1. HOW effectively do trained agents execute their learned strategies?
2. WHAT environmental factors (object positions, availability) influence tactical execution?
3. WHEN do agents make optimal vs suboptimal tactical decisions and WHY?
4. HOW do agents adapt their trained strategies to this specific episode?
5. WHAT competitive and cooperative patterns emerge between trained policies?

### ANALYSIS FORMAT: COMPREHENSIVE RESEARCH-QUALITY ANALYSIS (6000-10000 words recommended)
**REQUIRED STRUCTURE**: Executive Summary → Timeline Pattern Analysis → Key Agent Analysis → 
Strategic Execution Quality → Environmental Adaptation → Tactical Insights

**DEPTH REQUIREMENTS**:
- Provide detailed breakdowns of agent strategic performance with specific metrics
- Include comprehensive numerical analysis and statistical correlations
- Analyze strategic execution quality in depth with examples
- Discuss environmental impact on tactical adaptation within this episode
- Generate actionable insights for RL researchers about policy effectiveness
- Use rich analytical language with detailed explanations and evidence
- Identify and explain optimal execution moments and tactical failures
- Provide comparative analysis between high and low performing agents
- Include spatial strategy analysis and resource utilization patterns
- Discuss tactical adaptation dynamics within the episode

**ANALYSIS STYLE**: Emulate comprehensive academic research analysis with detailed insights,
specific agent examples, numerical evidence, and thorough exploration of all strategic dynamics.
Focus on ALL significant execution patterns and provide exhaustive coverage of the episode.

### FORMATTING REQUIREMENTS:
- Use emojis liberally throughout the analysis to enhance readability and engagement
- Include rich visual formatting with bullet points, numbered lists, and clear hierarchies
- Provide comprehensive data breakdowns with specific metrics for each section
- Use engaging headings and subheadings with emojis for visual appeal
- Include detailed comparative analysis sections with multiple agent examples
- Expand each major section to be comprehensive and thorough (aim for 500-800 words per section)

### EXAMPLE FORMATTING STYLE (emulate this engagement and depth):
🎯 EXECUTIVE SUMMARY
🔍 sp.av.replay.probe.500 Analysis: Strategic Execution Assessment
📊 Success Rate Paradox
Counterintuitive Performance Correlation:
📈 Higher Action Success Rates (25-35%) → Lower Final Scores
💡 Critical Insight: High action success rates indicate conservative execution strategies...
🏆 KEY AGENT ANALYSIS
💎 Agent 16 (14.0 points) - Elite Strategic Executor
⚡ Agent 6 (9.0 points) - Efficient Tactical Performer
❌ Agents 8 & 19 (0.0 points) - Strategic Execution Failures"""

    def _extract_statistical_insights(self, replay_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive statistical insights using the stats analysis system."""
        try:
            # Initialize statistics extractor
            stats_extractor = StatsExtractor()

            # Extract structured episode statistics
            episode_stats = stats_extractor.extract_from_replay_data(replay_data)

            # Initialize analysis engines
            behavioral_engine = BehavioralAnalysisEngine()
            resource_analyzer = ResourceFlowAnalyzer()
            combat_analyzer = CombatInteractionAnalyzer()
            efficiency_scorer = BuildingEfficiencyScorer()
            phase_detector = StrategicPhaseDetector()

            # Create agent stats objects from extracted data
            agent_stats = []
            for agent_data in episode_stats.get("agent", []):
                try:
                    agent_stat = AgentStats(
                        agent_id=agent_data.get("agent_id", 0),
                        total_actions=agent_data.get("action_counts", {}),
                        action_success_rates=agent_data.get("success_rates", {}),
                        resource_flows=agent_data.get("resource_flows", {}),
                        movement_patterns=agent_data.get("movement_stats", {}),
                        combat_stats=agent_data.get("combat_stats", {}),
                        building_interactions=agent_data.get("building_interactions", {}),
                        efficiency_metrics=agent_data.get("efficiency_metrics", {}),
                    )
                    agent_stats.append(agent_stat)
                except Exception:
                    # Skip malformed agent data
                    continue

            # Create building stats objects from extracted data
            building_stats = []
            for building_data in episode_stats.get("converter", []):
                try:
                    building_stat = BuildingStats(
                        building_id=building_data.get("building_id", 0),
                        type_id=building_data.get("type_id", 0),
                        type_name=building_data.get("type_name", f"type_{building_data.get('type_id', 0)}"),
                        location=building_data.get("location", (0, 0)),
                        production_efficiency=building_data.get("production_efficiency", {}),
                        resource_flows=building_data.get("resource_flows", {}),
                        operational_stats=building_data.get("operational_stats", {}),
                        bottleneck_analysis=building_data.get("bottleneck_analysis", {}),
                    )
                    building_stats.append(building_stat)
                except Exception:
                    # Skip malformed building data
                    continue

            # Generate comprehensive analysis
            insights = {}

            if agent_stats:
                try:
                    insights["behavioral_analysis"] = behavioral_engine.analyze_agent_behaviors(agent_stats)
                except Exception:
                    insights["behavioral_analysis"] = {}

                if building_stats:
                    try:
                        insights["resource_flow_analysis"] = resource_analyzer.analyze_resource_flows(
                            agent_stats, building_stats
                        )
                    except Exception:
                        insights["resource_flow_analysis"] = {}

                try:
                    insights["combat_analysis"] = combat_analyzer.analyze_combat_interactions(agent_stats)
                except Exception:
                    insights["combat_analysis"] = {}

                try:
                    episode_length = replay_data.get("episode_length", replay_data.get("max_steps", 1000))
                    insights["strategic_phases"] = phase_detector.detect_strategic_phases(agent_stats, episode_length)
                except Exception:
                    insights["strategic_phases"] = []

            if building_stats:
                try:
                    insights["building_efficiency"] = efficiency_scorer.score_building_efficiency(building_stats)
                except Exception:
                    insights["building_efficiency"] = {}

            return insights

        except Exception:
            # Return empty insights if extraction fails - don't break the analysis
            return {}

    def _create_user_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create user prompt with replay analysis data."""
        episode_length = analysis.get("episode_length", 0)
        agents = analysis.get("agents", [])
        final_scores = analysis.get("final_scores", {})
        environment_info = analysis.get("environment_info", {})
        key_events = analysis.get("key_events", [])
        policy_training_info = analysis.get("policy_training_info", {})

        prompt_parts = [
            "Analyze this Metta AI gridworld gameplay replay:",
            "",
            "EPISODE OVERVIEW:",
            f"- Duration: {episode_length} steps (max ~1000)",
            f"- Agents: {len(agents)} multi-agent environment",
            f"- Map: {environment_info.get('map_size', 'unknown')} gridworld with resource conversion mechanics",
            f"- Format: {environment_info.get('format', 'unknown')} replay data",
        ]

        # Add policy training information
        if policy_training_info:
            is_trained = policy_training_info.get("is_trained")
            confidence = policy_training_info.get("training_confidence", "unknown")
            policy_source = policy_training_info.get("policy_source", "unknown")
            version_info = policy_training_info.get("version_info")
            reasoning = policy_training_info.get("reasoning", "")

            prompt_parts.extend(
                [
                    "",
                    "POLICY INFORMATION:",
                    f"- Policy Source: {policy_source}",
                ]
            )

            if version_info:
                prompt_parts.append(f"- Version/Checkpoint: {version_info}")

            if is_trained is not None:
                training_status = "TRAINED" if is_trained else "UNTRAINED/EARLY TRAINING"
                prompt_parts.append(f"- Training Status: {training_status} (confidence: {confidence})")
                prompt_parts.append(f"- Reasoning: {reasoning}")
            else:
                prompt_parts.append(f"- Training Status: UNKNOWN (confidence: {confidence})")
                prompt_parts.append(f"- Reasoning: {reasoning}")
        else:
            prompt_parts.extend(
                [
                    "",
                    "POLICY INFORMATION:",
                    "- Policy Source: Not provided",
                    "- Training Status: UNKNOWN - analyze behaviors to determine if agents exhibit learned strategies",
                ]
            )

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
                    action_success_rate_val = float(behavior.get("action_success_rate", 0))
                except (ValueError, TypeError):
                    score_val = 0.0
                    distance_val = 0.0
                    action_success_rate_val = 0.0

                # Get strategic behavior
                strategic_behavior = behavior.get("strategic_behavior", "unknown")

                prompt_parts.append(
                    f"- {agent}: {score_val:.3f} points, {distance_val:.1f} units moved, "
                    f"{action_success_rate_val * 100:.2f}% action success, {strategic_behavior} strategy"
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
                other_resources = resource_objects.get("other_resources", [])

                # Break down all resource types separately for clear counting
                mine_types = {}
                generator_types = {}
                converter_types = {}
                other_types = {}

                for gen in generators:
                    type_name = gen.get("type_name", "unknown")
                    if "mine" in type_name:
                        if type_name not in mine_types:
                            mine_types[type_name] = 0
                        mine_types[type_name] += 1
                    else:
                        if type_name not in generator_types:
                            generator_types[type_name] = 0
                        generator_types[type_name] += 1

                for conv in converters:
                    type_name = conv.get("type_name", "unknown")
                    if type_name not in converter_types:
                        converter_types[type_name] = 0
                    converter_types[type_name] += 1

                for other in other_resources:
                    type_name = other.get("type_name", "unknown")
                    if type_name not in other_types:
                        other_types[type_name] = 0
                    other_types[type_name] += 1

                # Format the resource objects line with type breakdown
                # Fix pluralization for irregular words
                def get_plural(name, count):
                    if name == "factory":
                        return f"{count} factories"
                    elif name == "armory":
                        return f"{count} armories"
                    elif name == "lasery":
                        return f"{count} laseries"
                    else:
                        return f"{count} {name}s"

                # Build detailed breakdown listing each type individually
                resource_parts = []

                # List each mine type individually
                if mine_types:
                    for name, count in sorted(mine_types.items()):
                        resource_parts.append(get_plural(name, count))

                # List each generator type individually
                if generator_types:
                    for name, count in sorted(generator_types.items()):
                        resource_parts.append(get_plural(name, count))

                # Handle altars (may include temples)
                altar_type_counts = {}
                for altar in altars:
                    type_name = altar.get("type_name", "altar")
                    altar_type_counts[type_name] = altar_type_counts.get(type_name, 0) + 1

                for name, count in sorted(altar_type_counts.items()):
                    resource_parts.append(get_plural(name, count))

                # List each converter type individually (factories, labs, etc.)
                if converter_types:
                    for name, count in sorted(converter_types.items()):
                        resource_parts.append(get_plural(name, count))

                # List each other resource type individually (armories, laseries, etc.)
                if other_types:
                    for name, count in sorted(other_types.items()):
                        resource_parts.append(get_plural(name, count))

                if resource_parts:
                    prompt_parts.append(f"- Resource Objects: {', '.join(resource_parts)}")

                # Individual availability stats for each mine type
                if mine_types:
                    prompt_parts.append("")
                    prompt_parts.append("MINE AVAILABILITY:")
                    for mine_type in sorted(mine_types.keys()):
                        mine_objects_of_type = [g for g in generators if g.get("type_name") == mine_type]
                        if mine_objects_of_type:
                            try:
                                avg_hp = sum(float(m.get("hp", 0)) for m in mine_objects_of_type) / len(
                                    mine_objects_of_type
                                )
                            except (ValueError, TypeError):
                                avg_hp = 0.0
                            ready_count = len([m for m in mine_objects_of_type if m.get("cooldown_ready", False)])
                            prompt_parts.append(
                                f"- {mine_type}: {ready_count}/{len(mine_objects_of_type)} ready, avg HP: {avg_hp:.1f}"
                            )

                # Individual availability stats for each generator type
                if generator_types:
                    if not mine_types:  # Add header if mines section wasn't shown
                        prompt_parts.append("")
                    prompt_parts.append("GENERATOR AVAILABILITY:")
                    for gen_type in sorted(generator_types.keys()):
                        gen_objects_of_type = [g for g in generators if g.get("type_name") == gen_type]
                        if gen_objects_of_type:
                            try:
                                avg_hp = sum(float(g.get("hp", 0)) for g in gen_objects_of_type) / len(
                                    gen_objects_of_type
                                )
                            except (ValueError, TypeError):
                                avg_hp = 0.0
                            ready_count = len([g for g in gen_objects_of_type if g.get("cooldown_ready", False)])
                            prompt_parts.append(
                                f"- {gen_type}: {ready_count}/{len(gen_objects_of_type)} ready, avg HP: {avg_hp:.1f}"
                            )

                # Individual availability stats for altars/temples
                if altars:
                    prompt_parts.append("")
                    prompt_parts.append("ALTAR/TEMPLE AVAILABILITY:")
                    for altar_type in sorted(altar_type_counts.keys()):
                        altar_objects_of_type = [a for a in altars if a.get("type_name") == altar_type]
                        if altar_objects_of_type:
                            try:
                                avg_hp = sum(float(a.get("hp", 0)) for a in altar_objects_of_type) / len(
                                    altar_objects_of_type
                                )
                            except (ValueError, TypeError):
                                avg_hp = 0.0
                            ready_count = len([a for a in altar_objects_of_type if a.get("cooldown_ready", False)])
                            prompt_parts.append(
                                f"- {altar_type}: {ready_count}/{len(altar_objects_of_type)} ready, "
                                f"avg HP: {avg_hp:.1f}"
                            )

                # Individual availability stats for converters (factories, labs, etc.)
                if converter_types:
                    prompt_parts.append("")
                    prompt_parts.append("CONVERTER AVAILABILITY:")
                    for conv_type in sorted(converter_types.keys()):
                        conv_objects_of_type = [c for c in converters if c.get("type_name") == conv_type]
                        if conv_objects_of_type:
                            try:
                                avg_hp = sum(float(c.get("hp", 0)) for c in conv_objects_of_type) / len(
                                    conv_objects_of_type
                                )
                            except (ValueError, TypeError):
                                avg_hp = 0.0
                            ready_count = len([c for c in conv_objects_of_type if c.get("cooldown_ready", False)])
                            prompt_parts.append(
                                f"- {conv_type}: {ready_count}/{len(conv_objects_of_type)} ready, avg HP: {avg_hp:.1f}"
                            )

                # Individual availability stats for other resources (armories, laseries, etc.)
                if other_types:
                    prompt_parts.append("")
                    prompt_parts.append("OTHER RESOURCE AVAILABILITY:")
                    for other_type in sorted(other_types.keys()):
                        other_objects_of_type = [o for o in other_resources if o.get("type_name") == other_type]
                        if other_objects_of_type:
                            try:
                                avg_hp = sum(float(o.get("hp", 0)) for o in other_objects_of_type) / len(
                                    other_objects_of_type
                                )
                            except (ValueError, TypeError):
                                avg_hp = 0.0
                            ready_count = len([o for o in other_objects_of_type if o.get("cooldown_ready", False)])
                            prompt_parts.append(
                                f"- {other_type}: {ready_count}/{len(other_objects_of_type)} ready, "
                                f"avg HP: {avg_hp:.1f}"
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

                    # Show all checkpoints - no truncation
                    checkpoints_to_show = progression

                    for checkpoint_data in checkpoints_to_show:
                        # Safely extract and convert all numeric values
                        try:
                            step = int(checkpoint_data.get("step", 0))
                            score = float(checkpoint_data.get("score", 0))
                            distance = float(checkpoint_data.get("distance_traveled", 0))
                            action_success_rate = float(checkpoint_data.get("action_success_rate", 0))
                            action_count = int(checkpoint_data.get("action_count", 0))
                        except (ValueError, TypeError):
                            step, score, distance, action_success_rate, action_count = 0, 0.0, 0.0, 0.0, 0

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
                            f"  Step {step:4d}: {score:5.0f}pts, {distance:5.0f}units, "
                            f"{action_success_rate * 100:4.2f}% action success, pos({pos_r:3d},{pos_c:3d}), "
                            f"{action_count:4d} actions, {behavior} ({recent_action})"
                        )

                # Add summary if we truncated agents or checkpoints
                total_agents = len(agent_progression)
                if total_agents > 6:
                    prompt_parts.append(f"\n[Showing first 6 of {total_agents} agents - truncated for response size]")

        # Add action timeline visualization
        action_timelines = analysis.get("action_timelines", {})
        if action_timelines and "timelines" in action_timelines:
            prompt_parts.extend(
                [
                    "",
                    f"ACTION TIMELINES (first {action_timelines.get('max_steps_shown', 150)} steps):",
                ]
            )

            # Legend is provided in system prompt, not user prompt

            # Add timelines for each agent
            timelines_data = action_timelines.get("timelines", {})
            for agent_name, timeline_info in list(timelines_data.items()):  # Show all agent timelines
                timeline_str = timeline_info.get("timeline", "")
                action_counts = timeline_info.get("action_counts", {})
                total_actions = timeline_info.get("total_actions", 0)

                # Format action counts for display
                count_str = ", ".join([f"{char}:{count}" for char, count in action_counts.items()])

                prompt_parts.extend(
                    [
                        f"{agent_name}: {timeline_str}",
                        f"  → {total_actions} actions total ({count_str})",
                    ]
                )

        if analysis.get("analysis_error"):
            prompt_parts.extend(
                [
                    "",
                    f"NOTE: There was an error during data extraction: {analysis['analysis_error']}",
                    "Please provide analysis based on available data.",
                ]
            )

        # Add statistical insights from the stats analysis system
        statistical_insights = analysis.get("statistical_insights", {})
        if statistical_insights:
            prompt_parts.extend(
                [
                    "",
                    "STATISTICAL INSIGHTS:",
                ]
            )

            # Behavioral analysis insights
            behavioral_analysis = statistical_insights.get("behavioral_analysis", {})
            if behavioral_analysis:
                # Agent efficiency rankings with actual data
                efficiency_rankings = behavioral_analysis.get("efficiency_rankings", [])
                if efficiency_rankings:
                    # Show top 5 and bottom 5 performers
                    prompt_parts.append("- Agent Efficiency Rankings:")
                    for i, agent in enumerate(efficiency_rankings):
                        prompt_parts.append(
                            f"  #{i + 1}: Agent {agent.get('agent_id', 'unknown')} - "
                            f"efficiency {agent.get('efficiency_score', 0):.3f}"
                        )

                # Behavioral clusters with actual agent assignments
                behavioral_clusters = behavioral_analysis.get("behavioral_clusters", {})
                if behavioral_clusters:
                    prompt_parts.append("- Behavioral Clusters:")
                    for cluster_type, agent_list in behavioral_clusters.items():
                        if agent_list:
                            prompt_parts.append(f"  {cluster_type}: {len(agent_list)} agents ({agent_list})")

                # Strategy identification with agent mappings
                strategy_identification = behavioral_analysis.get("strategy_identification", {})
                if strategy_identification:
                    strategies_count = {}
                    for _, strategy in strategy_identification.items():
                        strategies_count[strategy] = strategies_count.get(strategy, 0) + 1

                    prompt_parts.append("- Agent Strategy Distribution:")
                    for strategy, count in sorted(strategies_count.items()):
                        prompt_parts.append(f"  {strategy}: {count} agents")

                # Performance correlations with actual values
                performance_correlations = behavioral_analysis.get("performance_correlations", {})
                if performance_correlations:
                    prompt_parts.append("- Performance Correlations:")
                    for metric_pair, correlation in performance_correlations.items():
                        prompt_parts.append(f"  {metric_pair}: {correlation:.3f}")

                # Outlier detection with details
                outliers = behavioral_analysis.get("outlier_detection", [])
                if outliers:
                    prompt_parts.append("- Behavioral Outliers:")
                    for outlier in outliers[:5]:  # Show first 5 outliers
                        agent_id = outlier.get("agent_id", "unknown")
                        reason = outlier.get("outlier_reason", "unusual behavior")
                        prompt_parts.append(f"  Agent {agent_id}: {reason}")

            # Resource flow analysis with detailed data
            resource_flow = statistical_insights.get("resource_flow_analysis", {})
            if resource_flow:
                # Resource scarcity analysis with actual values
                scarcity_analysis = resource_flow.get("resource_scarcity_analysis", {})
                if scarcity_analysis:
                    prompt_parts.append("- Resource Scarcity Analysis:")
                    for resource, scarcity_level in scarcity_analysis.items():
                        if isinstance(scarcity_level, (int, float)):
                            status = (
                                "abundant" if scarcity_level > 0.7 else "moderate" if scarcity_level > 0.3 else "scarce"
                            )
                            prompt_parts.append(f"  {resource}: {scarcity_level:.3f} ({status})")

                # Production efficiency with building details
                production_efficiency = resource_flow.get("production_efficiency", {})
                if production_efficiency:
                    prompt_parts.append("- Production Efficiency:")
                    for building, efficiency in production_efficiency.items():
                        if isinstance(efficiency, (int, float)):
                            prompt_parts.append(f"  {building}: {int(efficiency)}")

                # Bottleneck identification with specifics
                bottlenecks = resource_flow.get("bottleneck_identification", [])
                if bottlenecks:
                    prompt_parts.append("- Resource Bottlenecks:")
                    for bottleneck in bottlenecks[:5]:  # Show first 5 bottlenecks
                        location = bottleneck.get("location", "unknown")
                        resource = bottleneck.get("resource", "unknown")
                        severity = bottleneck.get("severity", 0)
                        prompt_parts.append(f"  {resource} at {location}: severity {severity:.2f}")

            # Combat analysis with detailed rankings
            combat_analysis = statistical_insights.get("combat_analysis", {})
            if combat_analysis:
                # Aggression rankings with actual scores
                aggression_rankings = combat_analysis.get("aggression_rankings", [])
                if aggression_rankings:
                    prompt_parts.append("- Combat Aggression Rankings:")
                    for i, agent_data in enumerate(aggression_rankings[:5]):
                        agent_id = agent_data.get("agent_id", "unknown")
                        score = agent_data.get("aggression_score", 0)
                        prompt_parts.append(f"  #{i + 1}: Agent {agent_id} - aggression {score:.3f}")

                # Cooperation metrics with details
                cooperation_metrics = combat_analysis.get("cooperation_metrics", {})
                if cooperation_metrics:
                    prompt_parts.append("- Cooperation Metrics:")
                    for metric, value in cooperation_metrics.items():
                        if isinstance(value, (int, float)):
                            prompt_parts.append(f"  {metric}: {value:.3f}")

            # Strategic phases with detailed timeline
            strategic_phases = statistical_insights.get("strategic_phases", [])
            if strategic_phases:
                prompt_parts.append("- Strategic Phase Timeline:")
                for phase in strategic_phases:
                    phase_num = phase.get("phase_number", 0)
                    start_step = phase.get("start_step", 0)
                    end_step = phase.get("end_step", 0)
                    strategy = phase.get("dominant_strategy", "unknown")
                    prompt_parts.append(f"  Phase {phase_num} (steps {start_step}-{end_step}): {strategy} strategy")

            # Building efficiency with detailed scores
            building_efficiency = statistical_insights.get("building_efficiency", {})
            if building_efficiency:
                individual_scores = building_efficiency.get("individual_scores", {})
                if individual_scores:
                    prompt_parts.append("- Building Efficiency Analysis:")
                    for building_id, score in individual_scores.items():
                        if isinstance(score, (int, float)):
                            prompt_parts.append(f"  Building {building_id}: efficiency {score:.3f}")

                # Optimization recommendations
                recommendations = building_efficiency.get("optimization_recommendations", [])
                if recommendations:
                    prompt_parts.append("- Building Optimization Recommendations:")
                    for rec in recommendations[:3]:  # Show first 3 recommendations
                        prompt_parts.append(f"  • {rec}")

        # Add a simple request for analysis (detailed instructions are now in system prompt)
        prompt_parts.extend(
            [
                "",
                "Please analyze this replay data according to the analysis requirements and "
                "formatting instructions provided in your system prompt.",
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
