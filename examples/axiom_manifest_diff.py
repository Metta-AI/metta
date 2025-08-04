#!/usr/bin/env uv run
"""Manifest diffing for ablation studies - track what changed."""

import json
import tempfile
from pathlib import Path

from metta.sweep.axiom import Ctx, Pipeline
from metta.sweep.axiom.manifest import diff_manifests


class ExperimentPipeline:
    """Pipeline that generates manifests."""
    
    def __init__(self, learning_rate: float, batch_size: int):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def build_pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .stage("configure", self._configure)
            .stage("train", self._train)
            .io("save_manifest", self._save_manifest)
        )
    
    def _configure(self):
        return {
            "config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "optimizer": "Adam",
            }
        }
    
    def _train(self, state):
        # Simulate training with different results based on config
        score = 0.8 + (self.learning_rate * 10) + (self.batch_size / 1000)
        return {**state, "score": score, "iterations": 1000}
    
    def _save_manifest(self, state):
        manifest = {
            "experiment": {
                "type": "training",
                "config": state["config"],
            },
            "results": {
                "score": state["score"],
                "iterations": state["iterations"],
            },
        }
        
        # Save to temp file
        temp_dir = Path(tempfile.gettempdir())
        manifest_file = temp_dir / f"manifest_{self.learning_rate}_{self.batch_size}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return {**state, "manifest": manifest, "manifest_path": str(manifest_file)}


def main():
    """Run experiments and compare manifests."""
    print("Running baseline experiment...")
    baseline = ExperimentPipeline(learning_rate=0.001, batch_size=32)
    baseline_result = baseline.build_pipeline().run(Ctx())
    
    print("Running variant experiment...")
    variant = ExperimentPipeline(learning_rate=0.01, batch_size=64)
    variant_result = variant.build_pipeline().run(Ctx())
    
    print("\n" + "=" * 50)
    print("Manifest Diff (what changed):")
    print("=" * 50)
    
    # Compare manifests
    diff = diff_manifests(
        baseline_result["manifest"],
        variant_result["manifest"],
        name1="baseline",
        name2="variant"
    )
    
    for line in diff:
        print(line)
    
    print("\n" + "=" * 50)
    print("Impact on results:")
    print(f"Baseline score: {baseline_result['score']:.4f}")
    print(f"Variant score: {variant_result['score']:.4f}")
    print(f"Improvement: {variant_result['score'] - baseline_result['score']:.4f}")


if __name__ == "__main__":
    main()