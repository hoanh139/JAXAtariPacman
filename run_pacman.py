#!/usr/bin/env python3
"""Run the Pacman game - interactive test script."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import jax
    import jax.numpy as jnp
    import jaxatari
    
    def main():
        """Main function to run Pacman game."""
        print("=" * 60)
        print("PACMAN GAME - JAXAtari")
        print("=" * 60)
        
        # Check if pacman is available
        available_games = jaxatari.list_available_games()
        print(f"\nAvailable games: {available_games}")
        
        if "pacman" not in available_games:
            print("ERROR: Pacman game not found in available games!")
            return
        
        print("\nCreating Pacman environment...")
        env = jaxatari.make("pacman")
        print(f"✓ Environment created")
        print(f"  Action space: {env.action_space()}")
        
        # Reset
        print("\nResetting environment...")
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        print(f"✓ Environment reset")
        print(f"  Initial score: {int(obs.score)}")
        print(f"  Initial lives: {int(obs.lives)}")
        print(f"  Player position: ({int(obs.player.x)}, {int(obs.player.y)})")
        print(f"  Pellets remaining: {int(jnp.sum(obs.pellets))}")
        print(f"  Power pellets remaining: {int(jnp.sum(obs.power_pellets))}")
        
        # Take some steps
        print("\n" + "=" * 60)
        print("Running game for 50 steps...")
        print("Actions: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN")
        print("=" * 60)
        
        total_reward = 0.0
        for step in range(50):
            # Simple strategy: try to move right, then up, then left, then down
            action = (step // 10) % 4 + 1  # Cycle through directions
            
            key, _ = jax.random.split(key)
            obs, state, reward, done, info = env.step(state, action)
            total_reward += float(reward)
            
            if step % 10 == 0 or done:
                print(f"Step {step:3d}: action={action}, reward={reward:6.2f}, "
                      f"score={int(obs.score):4d}, lives={int(obs.lives)}, "
                      f"pellets={int(jnp.sum(obs.pellets)):3d}, done={done}")
            
            if done:
                print(f"\nGame Over! Final score: {int(obs.score)}")
                break
        
        print(f"\nTotal reward: {total_reward:.2f}")
        print(f"Final score: {int(obs.score)}")
        print(f"Final lives: {int(obs.lives)}")
        print(f"Pellets remaining: {int(jnp.sum(obs.pellets))}")
        
        # Test rendering
        print("\n" + "=" * 60)
        print("Testing rendering...")
        try:
            image = env.render(state)
            print(f"✓ Rendering successful")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image value range: [{int(image.min())}, {int(image.max())}]")
        except Exception as e:
            print(f"✗ Rendering error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✓ Game test completed successfully!")
        print("=" * 60)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print("ERROR: Required dependencies not installed.")
    print(f"Missing: {e}")
    print("\nTo install dependencies, run:")
    print("  pip install -e .")
    print("\nOr install JAX first:")
    print("  pip install jax jaxlib")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


