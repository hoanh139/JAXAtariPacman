import jaxatari
import jax
import sys
import numpy as np

def test_mods():
    print("=== Testing Base PACMAN ===")
    base_env = jaxatari.make("pacman")
    obs, state = base_env.reset(jax.random.PRNGKey(42))
    print(f"Base Score Multipliers - Dot: {base_env.consts.PELLET_DOT_SCORE} | Fright Time: {base_env.consts.FRIGHTENED_DURATION}")

    print("\n=== Testing FasterPacmanMod (Score Multiplier) ===")
    mod_env = jaxatari.make("pacman", mods_config=["faster_pacman"])
    print(f"Modded Dot Score: {mod_env.consts.PELLET_DOT_SCORE} (Expected 12)")
    print(f"Modded Power Score: {mod_env.consts.PELLET_POWER_SCORE} (Expected 60)")
    
    print("\n=== Testing NoFrightMod ===")
    nf_env = jaxatari.make("pacman", mods_config=["no_fright"])
    print(f"Modded Fright Duration: {nf_env.consts.FRIGHTENED_DURATION} (Expected 0)")
    
    print("\n=== Testing HalfDotsMod ===")
    hd_env = jaxatari.make("pacman", mods_config=["half_dots"])
    _, hd_state = hd_env.reset(jax.random.PRNGKey(42))
    print(f"Base Start Dots: {state.dots_remaining} | HalfDots Start Dots: {hd_state.dots_remaining} (Expected ~120)")
    
    print("\n=== Testing RandomStartMod ===")
    rs_env = jaxatari.make("pacman", mods_config=["random_start"])
    _, rs_state = rs_env.reset(jax.random.PRNGKey(123))
    print(f"Base Start Pos: ({state.player_x}, {state.player_y}) | Random Start Pos: ({rs_state.player_x}, {rs_state.player_y})")

    print("\n=== Testing Coop Multiplayer Mod ===")
    coop_env = jaxatari.make("pacman", mods_config=["coop_multiplayer"])
    _, coop_state = coop_env.reset(jax.random.PRNGKey(42))
    print(f"Coop Player Config Array Shapes:")
    print(f"  - player_x={coop_state.player_x.shape}")
    print(f"  - player_y={coop_state.player_y.shape}")
    print(f"  - player_x values = {coop_state.player_x}")
    print(f"  - player_y values = {coop_state.player_y}")

if __name__ == "__main__":
    test_mods()
