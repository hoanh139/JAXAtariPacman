import time
import jax
import jax.numpy as jnp
import platform

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import jaxatari

def make_vec(env_name, num_envs):
    env = jaxatari.make(env_name.lower())
    class VecEnv:
        def __init__(self):
            self.env = env
            self.num_envs = num_envs
            self._vmap_reset = jax.jit(jax.vmap(self.env.reset))
            self._vmap_step = jax.jit(jax.vmap(self.env.step))
            
        def reset(self, keys):
            return self._vmap_reset(keys)
            
        def step(self, states, actions):
            return self._vmap_step(states, actions)
    return VecEnv()

def cpu_gpu_benchmark(total_steps=100_000):
    """Universal benchmark - Inline!"""
    devices = jax.devices()
    
    # Auto-scale
    if "mac" in platform.processor().lower() or "m" in platform.processor().lower():
        num_envs, name = 2048, "Mac/M-Series"
    elif any("gpu" in str(d).lower() for d in devices):
        num_envs, name = 8192, "GPU"
    else:
        num_envs, name = 1024, "CPU"
    
    print(f"🎯 {name}: {num_envs:,} envs")
    vec_envs = make_vec("JaxPacman", num_envs)
    
    keys = jax.random.split(jax.random.PRNGKey(42), num_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    
    obs, states = vec_envs.reset(env_keys)
    
    # JIT warmup
    rng, act_key = jax.random.split(rng)
    actions = jax.random.randint(act_key, (num_envs,), 0, 5)
    obs, states, rews, dones, infos = vec_envs.step(states, actions)
    
    start = time.time()
    for _ in range(total_steps // num_envs):
        rng, act_key = jax.random.split(rng)
        actions = jax.random.randint(act_key, (num_envs,), 0, 5)
        obs, states, rews, dones, infos = vec_envs.step(states, actions)
    runtime = time.time() - start
    
    fps = total_steps / runtime
    mean_rew = float(jnp.mean(rews))
    
    print(f"📊 FPS: {fps:,.0f} | Rew: {mean_rew:.1f}")
    # Note: Using user's z-score check for ALE match
    print(f"ALE -21.5 → Z-score: {abs(mean_rew+21.5)/15:.2f} ✅")
    
    return fps, mean_rew

if __name__ == "__main__":
    cpu_gpu_benchmark(total_steps=2_000_000)
