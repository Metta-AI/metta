# Failure Analysis: Scripted Agent

## Summary

Analyzed 4 failing environments (0/5 strategies succeed). Root causes identified:

1. **Navigation Bugs** (SINGLE_USE_WORLD, EXP2-EASY)
2. **Insufficient Exploration** (EXP1-HARD, GERMANIUM_CLUTCH)
3. **Extractor Discovery** (EXP1-HARD - no silicon found)
4. **Resource Gathering Logic** (EXP2-EASY - can't use discovered extractors)

---

## Detailed Failure Analysis

### 1. EXP1-HARD (0/5 strategies succeed)

**Symptoms** (200 steps observed):
- Hearts Assembled: 0
- Resources: Ge=5✅, Si=0❌, C=3/5⚠️, O=20✅
- Extractors Found: Ge=1, Si=0❌, C=1, O=1
- Phase: RECHARGE
- Energy: 52/100

**Root Causes**:
1. **❌ CRITICAL: No Silicon Extractors Found**
   - Agent explored but never discovered any silicon extractors
   - Without silicon, cannot assemble hearts
   - Exploration strategy not thorough enough for HARD difficulty

2. **⚠️ Insufficient Carbon**
   - Only found 1 carbon extractor, collected 3/5 needed
   - Likely depleted or on cooldown

3. **Limited Extractor Discovery**
   - Only found 3/4 resource types in 200 steps
   - HARD difficulty has fewer extractors, requires more thorough exploration

**Fix Priority**: HIGH
**Recommended Fixes**:
- Increase exploration duration for HARD difficulty (100 → 200+ steps)
- Implement "missing resource" detection: if a resource type has 0 extractors after N steps, force exploration
- Add bias toward unexplored map regions when critical resources missing

---

### 2. EXP2-EASY (0/5 strategies succeed)

**Symptoms** (200 steps observed):
- Hearts Assembled: 0
- Resources: Ge=0❌, Si=74✅, C=0❌, O=20✅
- Extractors Found: Ge=1, Si=1, C=1, O=3
- Phase: GATHER_CARBON
- Energy: 88/100

**Root Causes**:
1. **❌ CRITICAL: Navigation/Gathering Bug**
   - Agent found germanium and carbon extractors but collected 0 of each
   - Successfully collected 74 silicon (way more than needed!)
   - This suggests agent CAN'T REACH or CAN'T USE the germanium/carbon extractors

2. **Resource Gathering Logic Issue**
   - Agent is stuck in GATHER_CARBON phase but can't collect carbon
   - Likely the extractor is discovered but unreachable due to navigation bug

**Fix Priority**: CRITICAL
**Recommended Fixes**:
- Debug why agent can collect silicon (74!) but not germanium/carbon (0)
- Check if extractors are behind walls or in unreachable locations
- Verify navigation logic for reaching discovered extractors
- Add fallback: if stuck trying to gather a resource for N steps, mark as unobtainable and move on

---

### 3. SINGLE_USE_WORLD (0/5 strategies succeed)

**Symptoms** (200 steps observed):
- Hearts Assembled: 0
- Resources: Ge=2/5⚠️, Si=0❌, C=0❌, O=0❌
- Extractors Found: Ge=2, Si=2, C=1, O=1
- Phase: GATHER_GERMANIUM
- **❌ STUCK: 93 consecutive steps at same position**
- Energy: 100/100

**Root Causes**:
1. **❌ CRITICAL: Navigation Failure**
   - Agent stuck at same position for 93 steps
   - Full energy, so not an energy issue
   - Likely trying to reach a discovered extractor but pathfinding failing

2. **Single-Use Strategy Not Implemented**
   - Agent used 2 germanium extractors (max_uses=1 each)
   - Now trying to find more germanium but getting stuck
   - Needs "discovery-before-gathering" strategy for single-use worlds

**Fix Priority**: CRITICAL (navigation) + HIGH (strategy)
**Recommended Fixes**:
- **Navigation**: Fix pathfinding bug causing 93-step stuck loop
- **Strategy**: Implement single-use detection:
  - If all extractors have max_uses=1, enter "full discovery mode"
  - Explore entire map before gathering ANY resources
  - Count total extractors of each type before starting gathering

---

### 4. GERMANIUM_CLUTCH (0/5 strategies succeed)

**Symptoms** (200 steps observed):
- Hearts Assembled: 0
- Resources: Ge=4/5⚠️, Si=50✅, C=4/5⚠️, O=20✅
- Extractors Found: Ge=1❌, Si=2, C=3, O=3
- Phase: GATHER_GERMANIUM
- Energy: 83/100

**Root Causes**:
1. **❌ CRITICAL: Insufficient Germanium Extractors**
   - Only found 1 germanium extractor
   - Collected 4/5 germanium (likely depleted)
   - Need to find MORE germanium extractors

2. **⚠️ Carbon Also Low**
   - Collected 4/5 carbon from 3 extractors
   - May also be depleting

3. **Exploration Not Thorough Enough**
   - Agent stopped exploring too early
   - Needs to continue searching for germanium extractors

**Fix Priority**: MEDIUM
**Recommended Fixes**:
- Implement "resource shortage" detection:
  - If collected < required and all extractors depleted, force more exploration
  - Don't give up after initial exploration phase
- Add germanium-focused exploration bias when germanium is the limiting resource
- Increase exploration duration when critical resources are scarce

---

## Cross-Cutting Issues

### Issue 1: Navigation Bugs (affects 2/4 failures)
**Environments**: SINGLE_USE_WORLD, EXP2-EASY

**Evidence**:
- SINGLE_USE_WORLD: Stuck for 93 steps
- EXP2-EASY: Can't collect germanium/carbon despite finding extractors

**Impact**: CRITICAL - Blocks 50% of failures

**Root Cause**: Pathfinding fails to reach discovered extractors

**Fix**:
1. Debug Navigator class - why does it fail to reach known extractors?
2. Add "unreachable extractor" detection - if stuck for N steps, mark as unreachable
3. Implement fallback exploration when stuck

---

### Issue 2: Insufficient Exploration (affects 3/4 failures)
**Environments**: EXP1-HARD, GERMANIUM_CLUTCH, SINGLE_USE_WORLD

**Evidence**:
- EXP1-HARD: 0 silicon extractors found
- GERMANIUM_CLUTCH: Only 1 germanium extractor found
- SINGLE_USE_WORLD: Only 2 germanium extractors found (needs more)

**Impact**: HIGH - Blocks 75% of failures

**Root Cause**: Fixed exploration phase (100 steps) insufficient for:
- HARD difficulty (fewer extractors, larger search space)
- Critical resource shortages (need to find ALL extractors)
- Single-use worlds (must find everything before gathering)

**Fix**:
1. **Adaptive Exploration Duration**:
   ```python
   if difficulty == HARD:
       exploration_phase_steps = 200
   if mission_type == "single_use":
       exploration_phase_steps = 400  # Must find everything
   ```

2. **Continuous Exploration**:
   - Don't stop exploring after initial phase
   - If a resource type has 0 extractors found, force exploration
   - If all extractors of a type are depleted, search for more

3. **Resource-Focused Exploration**:
   - If germanium < 5 and all germanium extractors depleted, search for germanium
   - Bias exploration toward areas likely to have the missing resource

---

### Issue 3: Resource Gathering Logic (affects 1/4 failures)
**Environments**: EXP2-EASY

**Evidence**:
- Collected 74 silicon (7x more than needed!)
- Collected 0 germanium, 0 carbon (despite finding extractors)

**Impact**: MEDIUM - Specific to EXP2

**Root Cause**: Agent can reach some extractors but not others

**Fix**:
1. Verify all discovered extractors are actually reachable
2. Add "reachability test" when discovering extractors
3. Implement "stuck gathering" detection - if trying to gather for N steps with no progress, skip

---

## Recommended Fix Priority

### P0 (Critical - Blocks Multiple Failures)
1. **Fix Navigation Bugs**
   - Debug Navigator pathfinding failures
   - Add "unreachable" detection and fallback
   - **Impact**: Fixes SINGLE_USE_WORLD, EXP2-EASY

### P1 (High - Improves Exploration)
2. **Adaptive Exploration Duration**
   - Increase exploration for HARD difficulty (100 → 200 steps)
   - Implement "missing resource" detection → force exploration
   - **Impact**: Fixes EXP1-HARD, improves GERMANIUM_CLUTCH

3. **Continuous Exploration**
   - Don't stop after initial phase if resources missing
   - Re-explore when extractors depleted
   - **Impact**: Fixes GERMANIUM_CLUTCH, improves EXP1-HARD

### P2 (Medium - Specific Strategies)
4. **Single-Use World Strategy**
   - Detect max_uses=1 environments
   - Implement "discovery-before-gathering" mode
   - **Impact**: Fixes SINGLE_USE_WORLD

5. **Resource Shortage Detection**
   - Detect when collected < required and extractors depleted
   - Force exploration for that specific resource
   - **Impact**: Improves GERMANIUM_CLUTCH

---

## Expected Impact

**If P0 fixes implemented**:
- Current: 10/16 environments (62.5%)
- Expected: 12/16 environments (75%)
- Fixes: SINGLE_USE_WORLD, EXP2-EASY

**If P0 + P1 fixes implemented**:
- Expected: 14/16 environments (87.5%)
- Fixes: SINGLE_USE_WORLD, EXP2-EASY, EXP1-HARD, GERMANIUM_CLUTCH

**If all fixes implemented**:
- Expected: 15/16 environments (93.75%)
- Only remaining challenge: EXP2-HARD (complex map + hard difficulty)

