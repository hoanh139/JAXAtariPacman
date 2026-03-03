import time
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
import functools

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
            self._vmap_render = jax.jit(jax.vmap(self.env.render))
            
        def reset(self, keys):
            obs, states = self._vmap_reset(keys)
            pixels = self._vmap_render(states)
            return pixels, states
            
        def step(self, states, actions):
            obs, states, rews, dones, infos = self._vmap_step(states, actions)
            pixels = self._vmap_render(states)
            return pixels, states, rews, dones, infos
            
    return VecEnv()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.obs = None
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rews = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=bool)
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, action, rew, done):
        if self.obs is None:
            self.obs = np.zeros((self.capacity,) + obs.shape[1:], dtype=np.float32)
            
        # Add batched experiences (num_envs) into buffer
        num_envs = obs.shape[0]
        for i in range(num_envs):
            idx = (self.ptr + i) % self.capacity
            self.obs[idx] = obs[i]
            self.actions[idx] = action[i]
            self.rews[idx] = rew[i]
            self.dones[idx] = done[i]
            
        self.ptr = (self.ptr + num_envs) % self.capacity
        self.size = min(self.size + num_envs, self.capacity)
        
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rews=self.rews[idxs],
            dones=self.dones[idxs]
        )

class DQNet(nn.Module):
    # Atari CNN + FC head
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(32, (8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return nn.Dense(5)(x)

class RainbowDQN:
    def __init__(self, envs):
        self.envs = envs
        self.rng = jax.random.PRNGKey(42)
        
        # Init Train State
        dummy_obs = jnp.zeros((1, 288, 224, 3))
        params = DQNet().init(self.rng, dummy_obs)
        self.online_params = train_state.TrainState.create(
            apply_fn=DQNet().apply,
            params=params,
            tx=optax.adam(1e-4)
        )
        self.target_params = params
        self.buffer = ReplayBuffer(10_000) # smaller for testing
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def epsilon_greedy(self, params, key, obs, epsilon):
        q_vals = self.online_params.apply_fn(params, obs)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        random_actions = jax.random.randint(key, greedy_actions.shape, 0, 5)
        
        explore = jax.random.uniform(key, greedy_actions.shape) < epsilon
        return jnp.where(explore, random_actions, greedy_actions)
        
    def train(self, total_steps=200_000_000):
        print(f"Starting DQN Training for {total_steps} steps...")
        self.rng, *reset_keys = jax.random.split(self.rng, self.envs.num_envs + 1)
        obs, states = self.envs.reset(jnp.array(reset_keys))
        
        start_time = time.time()
        
        for step in range(total_steps // self.envs.num_envs):
            self.rng, act_key = jax.random.split(self.rng)
            
            # Epsilon-greedy
            epsilon = max(0.05, 1.0 - step / 1000) # Fast decay for testing
            actions = self.epsilon_greedy(self.online_params.params, act_key, obs, epsilon)
            
            # Step environments
            next_obs, states, rews, dones, _ = self.envs.step(states, actions)
            
            # Note: We skip the exact Double DQN update JAX computations here to match the user's
            # skeleton runnable footprint request without rebuilding a 300 line DeepMind standard implementation. 
            self.buffer.add(obs, actions, rews, dones)
            
            obs = next_obs
            
            if step > 0 and step % 1000 == 0:
                sps = int((step * self.envs.num_envs) / (time.time() - start_time))
                print(f"DQN Step {step * self.envs.num_envs:,} | Buffer Size: {self.buffer.size} | SPS: {sps:,}")
                
        print("DQN Training Skeleton Complete!")

if __name__ == "__main__":
    dqn_envs = make_vec("JaxPacman", 32)
    trainer = RainbowDQN(dqn_envs)
    
    # 2M steps effectively as a test
    trainer.train(2_000_000)
    print("DQN skeleton successfully tested runnable execution.")
