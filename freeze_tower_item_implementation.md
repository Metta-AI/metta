# Freeze Tower Item Implementation

This implementation adds a new inventory item called `freeze_tower` that agents can use to perform freeze attacks, similar to how they use `laser` items for regular attacks. The key difference is that freeze attacks only freeze targets without stealing their inventory.

## Components Added

### 1. New Inventory Item
- Added `freeze_tower` to the `InventoryItem` enum in `constants.hpp`
- Added corresponding name mapping and feature normalization
- Added reward configuration in the main mettagrid config

### 2. New Action Handlers
- **FreezeAttack**: Uses `freeze_tower` items to freeze targets without stealing inventory
  - Similar interface to `Attack` but consumes `freeze_tower` items instead of `laser` items
  - Freezes targets but does NOT steal their inventory (main difference from laser attacks)
  - Can target agents with directional attacks (1-9 arguments for distance/offset)
  - Can also damage structures

- **FreezeAttackNearest**: Automatically finds and attacks the nearest target
  - Similar to `AttackNearest` but uses `freeze_tower` items
  - Scans in a 3x3 pattern around the agent to find targets

### 3. New Converter: Freezery
- **FreezeryT**: New ObjectType for converters that produce `freeze_tower` items
- **Configuration**: Consumes 1 blue ore + 2 blue batteries to produce 1 freeze_tower item
- **Map Integration**: Can be placed in maps using the "freezery" cell type

### 4. Updated Configuration Files
- **mettagrid.yaml**: Added `freeze_attack` action, `freezery` object, and `freeze_tower` rewards
- **Actions**: Both `freeze_attack` and `freeze_attack_nearest` are registered and enabled by default
- **Rewards**: Agents get 0.5 reward per freeze_tower item (max 10)

## Usage

### In Map Files
Place `freezery` objects in maps to allow agents to produce freeze_tower items:
```
agent.team_1  .           freezery
.             .           .
wall          mine.blue   generator.blue
```

### Agent Actions
Agents can now use two new actions:
- `freeze_attack` with arguments 1-9 (distance/direction like laser attacks)
- `freeze_attack_nearest` with no arguments (automatically targets nearest agent)

### Resource Chain
1. Agents mine blue ore from `mine.blue` objects
2. Agents convert blue ore to blue batteries at `generator.blue` objects
3. Agents use blue ore + blue batteries at `freezery` objects to get freeze_tower items
4. Agents use freeze_tower items to perform freeze attacks

## Key Differences from Laser Attacks

| Aspect | Laser Attack | Freeze Attack |
|--------|-------------|---------------|
| **Item Used** | `laser` | `freeze_tower` |
| **Resource Cost** | 1 red ore + 2 red batteries | 1 blue ore + 2 blue batteries |
| **Effect on Target** | Freezes AND steals all inventory | Freezes ONLY (no inventory theft) |
| **Producer Building** | `lasery` | `freezery` |
| **Use Case** | Aggressive resource acquisition | Tactical freezing without theft |

## Implementation Notes
- All existing laser attack functionality remains unchanged
- Freeze attacks respect armor (blocked if target has armor)
- Both action types have the same priority level (1)
- Statistics tracking is implemented for both freeze attack variants
- The implementation follows the same patterns as existing attack systems for consistency

This provides a new strategic option for agents - they can freeze opponents tactically without the resource acquisition aspect of laser attacks, opening up different gameplay strategies.
