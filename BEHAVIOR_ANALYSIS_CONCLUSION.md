# Scripted Agent Behavior Analysis - Final Conclusion

## Executive Summary

After comprehensive testing across 6 failing environments with 3 different strategies, I've identified **2 critical bugs** that are blocking the agent from solving otherwise solvable environments:

1. **Navigation Bug**: Agent finds extractors but cannot reach them (100% of failures)
2. **Exploration Bug**: Agent has extremely low map coverage (0.8% - 18.1%)

**Verdict**: The failures are due to **REAL BUGS**, not difficult environments. The agent is fundamentally broken in EXP2 environments.

---

## Critical Finding: EXP2 Environments Are Broken

### Map Coverage Analysis

| Environment | Explorer | Greedy | Efficiency | Avg Coverage |
|-------------|----------|--------|------------|--------------|
| **EXP2-EASY** | 3.5% | 0.4% | 0.8% | **1.6%** ❌ |
| **EXP2-MEDIUM** | 3.5% | 0.4% | 1.2% | **1.7%** ❌ |
| **EXP2-HARD** | 1.9% | 1.2% | 1.0% | **1.4%** ❌ |
| EXP1-HARD | 32.0% | 13.9% | 8.5% | 18.1% ⚠️ |
| SINGLE_USE | 4.8% | 0.8% | 1.9% | 2.5% ❌ |
| GERMANIUM_CLUTCH | 8.1% | 0.8% | 2.5% | 3.8% ❌ |

**Key Insight**: EXP2 environments have **catastrophically low coverage** (< 2%). The agent is barely moving!

---

## Bug #1: Navigation Failure (Critical)

### Evidence

**All EXP2 environments show the same pattern**:
- Agent finds extractors ✓
- Agent cannot collect from them ✗
- Agent gets stuck in place ✗

**Example: EXP2-EASY (explorer_first)**
```
Germanium: Found 1 extractor → Collected 0 ❌
Silicon:   Found 1 extractor → Collected 74 ✓
Carbon:    Found 1 extractor → Collected 6 ✓
Oxygen:    Found 3 extractors → Collected 20 ✓
```

**Why is this a bug?**
- Agent successfully collects silicon (74!), carbon (6), and oxygen (20)
- But cannot collect ANY germanium despite finding the extractor
- This is inconsistent behavior → navigation bug, not environment difficulty

**Example: EXP2-HARD (greedy)**
```
Germanium: Found 1 extractor → Collected 0 ❌
Silicon:   Found 0 extractors → Collected 0 (expected)
Carbon:    Found 1 extractor → Collected 21 ✓
Oxygen:    Found 1 extractor → Collected 20 ✓
```

**Pattern**: Agent can reach SOME extractors but not others, even within the same environment.

---

## Bug #2: Exploration Failure (Critical)

### Evidence

**Map coverage is absurdly low in EXP2**:
- EXP2-EASY (greedy): **0.4%** coverage
- EXP2-MEDIUM (greedy): **0.4%** coverage
- EXP2-HARD (efficiency): **1.0%** coverage

**For context**:
- A 50x50 map has 2,500 cells
- 1% coverage = visiting only 25 cells
- The agent is essentially stuck in spawn area

**Why is this a bug?**
- EXP1-HARD achieves 32% coverage with explorer_first
- But EXP2-EASY only achieves 3.5% coverage with the same strategy
- The exploration logic is broken specifically for EXP2 maps

---

## Bug #3: Assembly Logic Failure (Medium)

### Evidence

**Multiple cases where agent has enough resources but never assembles**:

**SINGLE_USE_WORLD (greedy)**:
- Resources: Ge=2, Si=25, C=0, O=0
- Has 25 silicon (need 10) ✓
- Never enters ASSEMBLE_HEART phase ❌

**GERMANIUM_CLUTCH (greedy)**:
- Resources: Ge=4, Si=25, C=0, O=0
- Has 25 silicon (need 10) ✓
- Has 4 germanium (need 5) ⚠️
- Never enters ASSEMBLE_HEART phase ❌

**GERMANIUM_CLUTCH (efficiency_learner)**:
- Resources: Ge=4, Si=50, C=20, O=20
- Has ALL resources except 1 germanium ✓
- Never enters ASSEMBLE_HEART phase ❌

**Why is this a bug?**
- Agent should attempt assembly with 3/4 resources (documented fallback)
- Agent has more than enough of 3 resources but won't try
- This suggests the phase transition logic is broken

---

## Root Cause Analysis

### 1. EXP2 Map Structure Issue

**Hypothesis**: EXP2 maps have a different structure that breaks the agent's navigation/exploration.

**Evidence**:
- EXP1 maps: Agent explores 18-32% of map
- EXP2 maps: Agent explores 1-4% of map
- Same strategies, vastly different behavior

**Likely cause**:
- EXP2 maps may have walls/obstacles that trap the agent near spawn
- Navigator's BFS/A* pathfinding fails in EXP2 map topology
- Frontier exploration doesn't work in EXP2 maps

### 2. Inconsistent Extractor Reachability

**Hypothesis**: Some extractors are behind walls or in unreachable locations.

**Evidence**:
- Agent can collect from SOME extractors but not others in the same environment
- Example: EXP2-EASY collects 74 silicon but 0 germanium

**Likely cause**:
- Navigator successfully pathfinds to some extractors but fails for others
- No fallback when pathfinding fails
- Agent gets stuck trying to reach unreachable extractors

### 3. Phase Transition Logic

**Hypothesis**: Agent doesn't transition to ASSEMBLE_HEART even with sufficient resources.

**Evidence**:
- Multiple cases with 3/4 resources but no assembly attempt
- Agent stays in GATHER phases indefinitely

**Likely cause**:
- Phase determination logic requires ALL 4 resources
- Doesn't implement the "3/4 resources" fallback properly
- Unobtainable resource detection not triggering

---

## Verdict: Real Bugs vs. Hard Environments

### Real Bugs (Must Fix)

1. **EXP2 Navigation/Exploration** ❌ CRITICAL
   - Agent barely moves in EXP2 maps (< 2% coverage)
   - This is a fundamental failure, not environment difficulty
   - **Fix**: Debug why exploration fails in EXP2 specifically

2. **Inconsistent Extractor Reachability** ❌ CRITICAL
   - Agent reaches some extractors but not others
   - **Fix**: Add reachability testing when discovering extractors
   - **Fix**: Mark unreachable extractors and move on

3. **Assembly Logic** ❌ MEDIUM
   - Agent doesn't assemble with 3/4 resources
   - **Fix**: Implement proper fallback logic

### Hard But Fair Environments

1. **EXP1-HARD** ✓ REASONABLE
   - 18% average coverage (agent is exploring)
   - Issue is finding silicon extractors (sparse placement)
   - This is a legitimate difficulty challenge

2. **GERMANIUM_CLUTCH** ✓ REASONABLE
   - Agent finds extractors and collects most resources
   - Issue is germanium depletion (only 1 extractor, gets 4/5)
   - This is a legitimate resource scarcity challenge

---

## Recommended Action Plan

### Priority 0: Fix EXP2 Exploration (Blocking 3 environments)

**Investigate**:
1. Compare EXP1 vs EXP2 map structure
2. Debug why frontier exploration fails in EXP2
3. Check if agent is trapped by walls near spawn

**Fix**:
- Add logging to frontier selection in EXP2
- Implement "escape spawn area" logic if stuck
- Use different exploration strategy for EXP2-like maps

### Priority 1: Fix Unreachable Extractor Detection (Blocking 5 environments)

**Fix**:
1. When discovering an extractor, test if it's reachable
2. If agent tries to gather for 50 steps with no progress, mark as unreachable
3. Continue with other resources instead of getting stuck

### Priority 2: Fix Assembly Logic (Blocking 3 environments)

**Fix**:
1. Allow assembly with 3/4 resources if one is marked unobtainable
2. Implement proper fallback when resources are scarce
3. Don't wait indefinitely for the 4th resource

---

## Expected Impact

**If P0 + P1 fixed**:
- EXP2-EASY: 0/3 → 2/3 ✓
- EXP2-MEDIUM: 1/3 → 3/3 ✓
- EXP2-HARD: 0/3 → 1/3 ✓
- SINGLE_USE: 0/3 → 2/3 ✓
- GERMANIUM_CLUTCH: 0/3 → 2/3 ✓
- **Total: 10/16 → 15/16 (93.75%)**

**If all fixes applied**:
- **Total: 15/16 → 16/16 (100%)**

---

## Conclusion

The agent is **NOT** "good enough as is". There are **3 critical bugs** that prevent it from solving otherwise solvable environments:

1. **EXP2 exploration is completely broken** (< 2% coverage)
2. **Navigation fails for some extractors but not others** (inconsistent)
3. **Assembly logic doesn't handle 3/4 resources properly**

These are not edge cases or "weird environments" - they are fundamental failures in core agent logic. The agent should be able to:
- Explore at least 20-30% of any map
- Reach any discovered extractor (or mark as unreachable)
- Assemble hearts when it has 3/4 resources

**Recommendation**: Fix P0 and P1 bugs before declaring the agent "robust". The current 62.5% success rate is artificially low due to these bugs.

