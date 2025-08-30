#!/usr/bin/env python3
"""Demo script showing full RL-focused replay analysis."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from metta.mcp_server.llm_client import LLMClient
from metta.mcp_server.training_utils import _analyze_replay_data, _load_replay_file


async def demo_analysis(replay_file: str):
    """Demonstrate full replay analysis with RL insights."""
    print("=" * 80)
    print("METTA AI REPLAY ANALYSIS DEMO")
    print("=" * 80)

    # Load and analyze replay data
    print("Step 1: Loading replay file...")
    replay_data = _load_replay_file(replay_file)

    print("Step 2: Analyzing replay data...")
    analysis = _analyze_replay_data(replay_data)

    print("Step 3: Extracting key metrics...")
    print(f"Episode Length: {analysis.get('episode_length', 0)} steps")
    print(f"Number of Agents: {len(analysis.get('agents', []))}")

    final_scores = analysis.get("final_scores", {})
    if final_scores:
        top_performers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 Performers:")
        for agent, score in top_performers:
            print(f"  {agent}: {score:.1f} points")

    # Check if ASCII map was generated
    ascii_map = analysis.get("environment_info", {}).get("ascii_map")
    if ascii_map and "ASCII rendering failed" not in ascii_map:
        print("✓ ASCII map generated successfully")
        map_lines = ascii_map.split("\n")
        print(f"  Map size: {len(map_lines)} lines")
    else:
        print("✗ ASCII map generation failed")

    print("\nStep 4: Generating Claude LLM analysis...")

    # Generate LLM analysis
    try:
        client = LLMClient()
        summary = await client.generate_replay_summary(analysis)

        print("=" * 80)
        print("CLAUDE AI ANALYSIS")
        print("=" * 80)
        print(summary)

    except Exception as e:
        print(f"LLM analysis failed: {e}")
        print("\nBut here's the analysis data structure that would be sent:")
        user_prompt = client._create_user_prompt(analysis)
        print("=" * 80)
        print("USER PROMPT (first 2000 chars):")
        print("=" * 80)
        print(user_prompt[:2000])
        if len(user_prompt) > 2000:
            print(f"\n... [truncated, full prompt is {len(user_prompt)} characters]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo RL-focused replay analysis")
    parser.add_argument("replay_file", help="Path to replay JSON file")
    args = parser.parse_args()

    # Run the async demo
    asyncio.run(demo_analysis(args.replay_file))
