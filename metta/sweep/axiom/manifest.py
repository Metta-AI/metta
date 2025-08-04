"""Manifest generation and diffing for experiment tracking."""

from typing import Any, Optional


def diff_manifests(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    ignore_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Compute diff between two experiment manifests.
    
    Args:
        baseline: Baseline manifest
        variant: Variant manifest
        ignore_keys: Keys to ignore in diff
    
    Returns:
        Dictionary of differences grouped by section
    """
    ignore_keys = ignore_keys or {"timestamp", "duration_seconds", "pipeline_result"}
    diff = {}
    
    # Recursively find differences
    def _diff_recursive(b: Any, v: Any, path: str = "") -> Optional[dict]:
        if isinstance(b, dict) and isinstance(v, dict):
            changes = {}
            all_keys = set(b.keys()) | set(v.keys())
            
            for key in all_keys:
                if key in ignore_keys:
                    continue
                
                full_path = f"{path}.{key}" if path else key
                
                if key not in b:
                    changes[key] = {"added": v[key]}
                elif key not in v:
                    changes[key] = {"removed": b[key]}
                else:
                    sub_diff = _diff_recursive(b[key], v[key], full_path)
                    if sub_diff:
                        changes[key] = sub_diff
            
            return changes if changes else None
        
        elif isinstance(b, list) and isinstance(v, list):
            if b != v:
                return {"baseline": b, "variant": v}
            return None
        
        else:
            if b != v:
                return {"baseline": b, "variant": v}
            return None
    
    # Compute top-level differences
    result = _diff_recursive(baseline, variant)
    
    # Group by semantic categories
    if result:
        # Extract specific sections
        sections = [
            "controls",
            "fingerprints",
            "environment",
            "hardware",
            "code",
            "joins",
        ]
        
        for section in sections:
            if section in result:
                diff[section] = result[section]
        
        # Everything else goes into "other"
        other = {k: v for k, v in result.items() if k not in sections}
        if other:
            diff["other"] = other
    
    return diff


def format_manifest_diff(diff: dict[str, Any]) -> str:
    """Format manifest diff for human consumption.
    
    Args:
        diff: Diff dictionary from diff_manifests
    
    Returns:
        Formatted string for display
    """
    lines = []
    
    def _format_value(v: Any, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        if isinstance(v, dict):
            result = []
            for key, val in v.items():
                if key in ["baseline", "variant"]:
                    result.append(f"{prefix}{key}: {val}")
                elif key in ["added", "removed"]:
                    result.append(f"{prefix}{key}: {val}")
                else:
                    result.append(f"{prefix}{key}:")
                    result.extend(_format_value(val, indent + 1))
            return result
        else:
            return [f"{prefix}{v}"]
    
    # Format each section
    for section, changes in diff.items():
        lines.append(f"\n[{section.upper()}]")
        lines.extend(_format_value(changes, 1))
    
    return "\n".join(lines)


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Create a summary of key manifest fields.
    
    Args:
        manifest: Full experiment manifest
    
    Returns:
        Summary dictionary with key information
    """
    summary = {
        "experiment": manifest.get("experiment"),
        "tag": manifest.get("tag"),
        "seed": manifest.get("controls", {}).get("seed"),
        "deterministic": manifest.get("controls", {}).get("enforce_determinism"),
        "platform": manifest.get("environment", {}).get("platform"),
        "torch_version": manifest.get("environment", {}).get("torch"),
        "cuda": manifest.get("environment", {}).get("cuda"),
        "gpu_count": manifest.get("hardware", {}).get("gpu_count", 0),
        "joins_provided": manifest.get("joins", {}).get("provided", []),
        "duration_seconds": manifest.get("duration_seconds"),
    }
    
    # Add fingerprints if present
    fingerprints = manifest.get("fingerprints", {})
    if fingerprints.get("dataset"):
        summary["dataset_hash"] = fingerprints["dataset"][:8]
    if fingerprints.get("env"):
        summary["env_hash"] = fingerprints["env"][:8]
    
    # Add git info if available
    code = manifest.get("code", {})
    if code.get("git_commit"):
        summary["git_commit"] = code["git_commit"][:8]
    if code.get("has_uncommitted_changes"):
        summary["uncommitted_changes"] = True
    
    return summary