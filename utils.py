"""Utility helpers for running and visualising trained genomes."""

from typing import Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import imageio

from genome import Genome
from slimevolly import SlimeVolleyTask
from evojax.task.slimevolley import SlimeVolley
from functools import partial

def logits_to_buttons(logits: Sequence[float]) -> np.ndarray:
    """Convert action logits (length 8) to binary button array (left,right,jump)."""
    idx = int(np.argmax(np.asarray(logits)))
    return np.array([(idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1], dtype=np.float32)


def rollout_frames(
    genome: Genome,
    steps: int = 400,
    frame_skip: int = 5,
    seed: int = 0,
    env: Optional[SlimeVolley] = None,
):
    """Roll a genome in the environment and capture frames."""

    if env is None:
        env = SlimeVolley(max_steps=1000, test=True)
    fwd = genome.forward_jax()
    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), 1))

    frames = []
    for step in range(steps):
        obs = state.obs[0]
        logits = fwd(jnp.asarray(obs))
        action = logits_to_buttons(logits)[None, :]
        state, _, _ = env.step(state, action)
        if step % frame_skip == 0:
            s0 = jax.tree_util.tree_map(lambda x: x[0], state)
            frames.append(np.array(SlimeVolley.render(s0)))
    return frames


def save_gif(frames, filename: str, fps: int = 20, scale: int = 2) -> None:
    """Save a list of RGB frames to a GIF file."""

    if scale != 1:
        frames = [np.kron(f, np.ones((scale, scale, 1), dtype=f.dtype)) for f in frames]
    imageio.mimsave(filename, frames, duration=int(1000 / fps))

# ---- one game between two parameter sets --------------------------
@partial(jax.jit, static_argnums=0)
def _play_match(policy_fn, params_a, params_b, rng):
    env   = SlimeVolley(max_steps=1000, test=False)      # no built-in bot
    state = env.reset(rng)
    pst_a = pst_b = None
    reward = 0.0

    for _ in range(env.max_steps):
        obs_a, obs_b = state.obs[:, 0], state.obs[:, 1]  # left/right views
        act_a, pst_a = policy_fn(params_a, obs_a, pst_a)
        act_b, pst_b = policy_fn(params_b, obs_b, pst_b)
        actions      = jnp.stack([act_a, act_b], 1)      # (B,2,3)

        state, raw_r, done, _ = env.step(state, actions)

        # shaped reward: 10 × score diff  + 0.01 per frame alive
        reward += 10.0 * raw_r[:, 0] + 0.01

    return reward                                        # scalar per batch

# ---- population-level fitness -------------------------------------
def selfplay_eval(pop_params, policy_fn, rng):
    """returns fitness array  len = len(pop_params)"""
    opp_idx = jax.random.permutation(rng, len(pop_params))
    keys    = jax.random.split(rng, len(pop_params))
    vmatch  = jax.vmap(_play_match, in_axes=(None,0,0,0))
    return vmatch(policy_fn, pop_params, pop_params[opp_idx], keys)

__all__ = ["logits_to_buttons", "rollout_frames", "save_gif"]
