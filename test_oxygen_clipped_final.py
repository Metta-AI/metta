import logging
from cogames.cogs_vs_clips.eval_missions import ClipOxygen
from cogames.policy.hyperparameter_presets import HYPERPARAMETER_PRESETS
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_oxygen_clipped_final():
    print("🚀 Testing oxygen_clipped evaluation with subprocess (no old unclipping logic)...")
    
    try:
        mission = ClipOxygen()
        mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
        env_config = mission.make_env()
        env_config.game.max_steps = 200  # Shorter run for testing

        env = MettaGridEnv(env_config)
        hyperparams_obj = HYPERPARAMETER_PRESETS["explorer"]
        policy = ScriptedAgentPolicy(env, hyperparams=hyperparams_obj)

        obs, info = env.reset()
        policy.reset(obs, info)
        agent_policy = policy.agent_policy(0)

        print("\n🎯 Starting test with subprocess only...")
        for step in range(env_config.game.max_steps):
            action = agent_policy.step(obs[0])
            obs, rewards, dones, truncated, info = env.step([action])
            
            agent_state = agent_policy._state
            
            # Log key state changes
            if step % 20 == 0 or agent_state.current_phase.name.startswith("GATHER_") or "UnclipSubprocess" in str(agent_state):
                print(f"Step {step:3d}: Phase={agent_state.current_phase.name}, "
                      f"C={agent_state.carbon}, O={agent_state.oxygen}, D={agent_state.decoder}, "
                      f"E={agent_state.energy}, H={agent_state.heart}")

            if dones[0] or truncated[0]:
                break

        # Final results
        agent_state = agent_policy._state
        print(f"\n📊 FINAL RESULTS:")
        print(f"  Hearts assembled: {agent_state.hearts_assembled}")
        print(f"  Final inventory: C={agent_state.carbon}, O={agent_state.oxygen}, "
              f"G={agent_state.germanium}, Si={agent_state.silicon}, "
              f"E={agent_state.energy}, H={agent_state.heart}, D={agent_state.decoder}")
        
        success = agent_state.hearts_assembled > 0
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: {'Agent completed the mission!' if success else 'Agent did not complete the mission.'}")
        
        return success

    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    test_oxygen_clipped_final()
