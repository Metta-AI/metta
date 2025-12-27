import time
import jax
import jax.numpy as jnp
from mettagrid_jax.env import MettaGridJAX
from mettagrid_jax import env_types as types

def main():
    print("Initializing MettaGridJAX...")
    num_agents = 10000
    env = MettaGridJAX(types.EnvParams(num_agents=num_agents, map_width=128, map_height=128))

    key = jax.random.PRNGKey(42)
    print("Resetting environment...")
    state = env.reset(key)

    # Compile step
    print("Compiling step function...")
    step_fn = jax.jit(env.step)
    obs_fn = jax.jit(env.observation)

    # Warmup
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (num_agents,), 0, 5)
    state = step_fn(state, actions)
    obs = obs_fn(state)
    obs.block_until_ready()
    print("Compilation done.")

    # Benchmark
    steps = 100
    print(f"Running {steps} steps with {num_agents} agents...")

    start = time.time()

    current_state = state
    current_key = key

    # Using lax.scan for maximum throughput measurement (removes Python overhead)
    def loop_body(carry, _):
        s, k = carry
        k, sk = jax.random.split(k)
        act = jax.random.randint(sk, (num_agents,), 0, 5)
        s_next = env.step(s, act)
        o = env.observation(s_next)
        return (s_next, k), None

    loop_fn = jax.jit(lambda s, k: jax.lax.scan(loop_body, (s, k), None, length=steps))

    (final_state, final_key), _ = loop_fn(current_state, current_key)

    # Block
    final_state.agents.pos.block_until_ready()
    end = time.time()

    duration = end - start
    fps = steps / duration
    print(f"Finished in {duration:.4f}s")
    print(f"SPS (Steps Per Second): {fps:.2f}")
    print(f"Agent SPS: {fps * num_agents:.2f}")

if __name__ == "__main__":
    main()
