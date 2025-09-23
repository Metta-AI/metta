from __future__ import annotations

from typing import Optional

# Default verb aliasing for task entrypoints
DEFAULT_VERB_ALIASES: dict[str, list[str]] = {
    # non-remote
    "sim": ["evaluate", "sim", "eval"],
    "eval": ["evaluate", "sim", "eval"],
    "evaluate": ["evaluate", "sim", "eval"],
    # remote
    "sim_remote": ["evaluate_remote", "sim_remote", "eval_remote"],
    "eval_remote": ["evaluate_remote", "sim_remote", "eval_remote"],
    "evaluate_remote": ["evaluate_remote", "sim_remote", "eval_remote"],
}


def generate_candidate_paths(
    primary: Optional[str],
    second: Optional[str] = None,
    *,
    auto_prefixes: list[str] | None = None,
    short_only: bool = True,
    verb_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Generate ordered candidate import paths.

    - primary: main symbol path like "arena.train" or fully-qualified.
    - second: when provided, treats inputs like (x, y) as the sugar y.x.
    - auto_prefixes: optional module prefixes to try (e.g., ["experiments.recipes"]).
    - short_only: if True, only apply prefixes for short forms (<= 1 dot).
    """
    if not primary:
        return []

    bases: list[str] = []
    if second:
        # Avoid redundant or confusing two-token cases like "train train"
        # We still produce a candidate, but keep primary-as-is before prefixed variants.
        bases.append(f"{second}.{primary}")
    bases.append(primary)

    prefixes = auto_prefixes or []
    candidates: list[str] = []
    for base in bases:
        expanded: list[str] = [base]
        # Expand verb aliases if any
        if "." in base:
            module_name, verb = base.rsplit(".", 1)
            aliases = (verb_aliases or {}).get(verb)
            if aliases:
                expanded = [f"{module_name}.{v}" for v in aliases]
        for item in expanded:
            candidates.append(item)
            # Optionally try prefixed variants
            if (not short_only) or (item.count(".") <= 1):
                for pref in prefixes:
                    if not item.startswith(pref + "."):
                        candidates.append(f"{pref}.{item}")

    # Deduplicate preserving order
    ordered: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered
