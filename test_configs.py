#!/usr/bin/env python3
"""Test script to validate configurations work with the new protein implementation."""

from omegaconf import OmegaConf

from metta.rl.carbs.metta_protein import MettaProtein


def test_configs():
    print("Testing configurations with new Protein implementation...")

    # Test demo config
    print("1. Testing demo_sweep.yaml...")
    cfg = OmegaConf.load("configs/sweep/demo_sweep.yaml")
    protein = MettaProtein(cfg)
    suggestion, info = protein.suggest()
    print(f"   ✅ Demo config works - suggestion: {suggestion}")

    # Test minimal config
    print("2. Testing minimal_protein_sweep.yaml...")
    cfg = OmegaConf.load("configs/sweep/minimal_protein_sweep.yaml")
    protein = MettaProtein(cfg)
    suggestion, info = protein.suggest()
    print(f"   ✅ Minimal config works - suggestion: {suggestion}")

    print("🎉 All configurations validated successfully!")


if __name__ == "__main__":
    test_configs()
