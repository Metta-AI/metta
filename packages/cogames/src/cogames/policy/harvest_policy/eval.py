"""Debug evaluation script for harvest policy."""
import sys
sys.path.insert(0, '.')

from harvest.harvest_policy import HarvestPolicy, HarvestPhase

from cogames.cli.mission import get_mission
from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

# Load mission config  
mission_name, mission_cfg, _ = get_mission('training_facility.harvest')
print(f'Mission loaded: {mission_name}')
print(f'Episode length: {mission_cfg.game.max_steps}')
print(f'Num agents: {mission_cfg.game.num_agents}')

# Create policy env info
env_interface = PolicyEnvInterface.from_mg_cfg(mission_cfg)

# Create policy instances
policy = HarvestPolicy(env_interface)
print(f'Policy created: {type(policy).__name__}')

# Get heart recipe
for protocol in env_interface.assembler_protocols:
    if protocol.output_resources.get("heart", 0) > 0:
        print(f'Heart recipe: {dict(protocol.input_resources)}')
        break

# Create agent policies list
agent_policies = [policy.agent_policy(i) for i in range(mission_cfg.game.num_agents)]

# Create rollout
rollout = Rollout(
    mission_cfg,
    agent_policies,
    max_action_time_ms=250,
    seed=42,
)

# Debug tracking
hearts_deposited = 0
ge_extractions = 0
last_ge = 0
last_hearts = 0
phase_counts = {phase: 0 for phase in HarvestPhase}

print('\nRunning evaluation...\n')

max_steps = mission_cfg.game.max_steps
step = 0
while not rollout.is_done():
    rollout.step()
    step += 1
    
    # Get agent state for debug (agent 0)
    state = agent_policies[0]._state
    
    # Track phase time
    phase_counts[state.phase] += 1
    
    # Track germanium extractions
    if state.germanium > last_ge:
        ge_extractions += (state.germanium - last_ge)
        print(f'Step {step}: Extracted germanium! inv_ge={state.germanium}, Total Ge extractions: {ge_extractions}')
    last_ge = state.germanium
    
    # Track heart deposits (hearts go down when deposited)
    if state.hearts < last_hearts:
        hearts_deposited += (last_hearts - state.hearts)
        print(f'Step {step}: Deposited heart! Total hearts: {hearts_deposited}')
    last_hearts = state.hearts
    
    # Periodic status
    if step % 1000 == 0:
        # Check germanium extractor cooldowns
        ge_exts = state.extractors.get('germanium', [])
        ge_cd = ge_exts[0].cooldown_remaining if ge_exts else 'N/A'
        
        print(f'Step {step}: phase={state.phase.name}, pos=({state.row},{state.col}), '
              f'inv=C{state.carbon}/O{state.oxygen}/G{state.germanium}/S{state.silicon}/H{state.hearts}, '
              f'energy={state.energy}, Ge_cd={ge_cd}')

# Output stats
print('\n=== Final Results ===')
print(f'Steps: {step}')
print(f'Total Ge extractions: {ge_extractions}')
print(f'Total hearts deposited: {hearts_deposited}')
print(f'\nPhase distribution:')
total_steps = sum(phase_counts.values())
for phase, count in phase_counts.items():
    pct = 100 * count / total_steps if total_steps > 0 else 0
    print(f'  {phase.name}: {count} steps ({pct:.1f}%)')

# Check final agent state
state = agent_policies[0]._state
print(f'\nFinal agent state:')
print(f'  Position: ({state.row}, {state.col})')
print(f'  Inventory: C{state.carbon}/O{state.oxygen}/G{state.germanium}/S{state.silicon}/H{state.hearts}')
print(f'  Energy: {state.energy}')
print(f'  Extractors discovered:')
for resource, exts in state.extractors.items():
    for ext in exts:
        print(f'    {resource}: pos={ext.position}, cd={ext.cooldown_remaining}, uses={ext.remaining_uses}, clipped={ext.clipped}')
