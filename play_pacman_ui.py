#!/usr/bin/env python3
"""Play Pacman game with UI using pygame."""

import sys
import os
import pygame
import jax
import jax.random as jrandom
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import jaxatari
    from scripts.utils import update_pygame, get_human_action
    
    UPSCALE_FACTOR = 4
    
    def main():
        """Main function to run Pacman game with UI."""
        print("=" * 60)
        print("PACMAN GAME - JAXAtari UI")
        print("=" * 60)
        print("\nControls:")
        print("  Arrow Keys or WASD: Move")
        print("  P: Pause/Resume")
        print("  R: Reset game")
        print("  ESC: Quit")
        print("=" * 60)
        
        # Create environment
        print("\nLoading Pacman game...")
        try:
            env = jaxatari.make("pacman")
            print("✓ Game loaded successfully")
        except Exception as e:
            print(f"✗ Error loading game: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("JAXAtari - Pacman")
        
        # Initialize environment
        master_key = jrandom.PRNGKey(42)
        reset_counter = 0
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)
        jitted_render = jax.jit(env.render)
        
        # Reset environment
        reset_key = jrandom.fold_in(master_key, reset_counter)
        obs, state = jitted_reset(reset_key)
        reset_counter += 1
        
        # Get render shape
        image = jitted_render(state)
        env_render_shape = image.shape[:2]
        
        # Create window
        window = pygame.display.set_mode(
            (env_render_shape[1] * UPSCALE_FACTOR, env_render_shape[0] * UPSCALE_FACTOR)
        )
        clock = pygame.time.Clock()
        
        # Game state
        running = True
        pause = False
        frame_rate = 30
        total_return = 0.0
        
        print("\nStarting game...")
        print("Use arrow keys or WASD to move!")
        
        # Main game loop
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        continue
                    elif event.key == pygame.K_p:
                        pause = not pause
                        print(f"{'Paused' if pause else 'Resumed'}")
                    elif event.key == pygame.K_r:
                        reset_key = jrandom.fold_in(master_key, reset_counter)
                        obs, state = jitted_reset(reset_key)
                        reset_counter += 1
                        total_return = 0.0
                        print("Game reset!")
            
            if pause:
                # Still render when paused
                image = jitted_render(state)
                update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
                clock.tick(frame_rate)
                continue
            
            # Get action from keyboard
            try:
                action = get_human_action()
            except SystemExit:
                running = False
                continue
            
            # Map action to pacman actions (0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN)
            # The get_human_action returns JAXAtariAction constants, we need to map them
            action_int = int(action)
            
            # Map JAXAtariAction to Pacman actions
            # JAXAtariAction: NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5
            # Pacman uses: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN
            action_map = {
                0: 0,   # NOOP -> NOOP
                2: 1,   # UP -> UP
                3: 2,   # RIGHT -> RIGHT
                4: 3,   # LEFT -> LEFT
                5: 4,   # DOWN -> DOWN
            }
            pacman_action = action_map.get(action_int, 0)
            
            # Step environment
            obs, state, reward, done, info = jitted_step(state, pacman_action)
            total_return += float(reward)
            
            # Update window title with score
            score = int(obs.score) if hasattr(obs, 'score') else 0
            lives = int(obs.lives) if hasattr(obs, 'lives') else 0
            pygame.display.set_caption(
                f"JAXAtari - Pacman | Score: {score} | Lives: {lives} | Reward: {total_return:.1f}"
            )
            
            # Check if done
            if done:
                print(f"\nGame Over! Final Score: {score}, Total Reward: {total_return:.2f}")
                total_return = 0.0
                reset_key = jrandom.fold_in(master_key, reset_counter)
                obs, state = jitted_reset(reset_key)
                reset_counter += 1
            
            # Render
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)
        
        pygame.quit()
        print("\nGame closed. Thanks for playing!")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print("ERROR: Required dependencies not installed.")
    print(f"Missing: {e}")
    print("\nTo install dependencies, run:")
    print("  pip install -e .")
    print("\nOr install JAX and pygame:")
    print("  pip install jax jaxlib pygame")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

