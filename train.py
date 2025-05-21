"""Training utilities for NEAT on Slime Volley."""

from typing import Callable, List
import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, InnovationTable
from neat import NEAT
from slimevolly import SlimeVolleyTask

def _evaluate_factory(task: SlimeVolleyTask) -> Callable[[List[Genome]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return an evaluation function for the given task.

    The returned function evaluates a list of genomes and
    returns ``(fitness, scored, conceded)`` arrays.
    """

    pop_size = task.pop

    def evaluate(genomes: List[Genome]):
        fwd_fns = [g.forward_jax() for g in genomes]
        state = task.reset()
        fitness = np.zeros(pop_size, dtype=np.float32)
        scored = np.zeros(pop_size, dtype=np.int32)
        conceded = np.zeros(pop_size, dtype=np.int32)
        done = jnp.zeros(pop_size, dtype=bool)
        while not bool(done.all()):
            obs = jax.vmap(lambda s: s.obs)(state)
            logits = jnp.stack([f(obs[i]) for i, f in enumerate(fwd_fns)])
            acts = (
                (jnp.arange(3) & (jnp.argmax(logits, 1)[:, None] >> jnp.arange(3)))
                > 0
            ).astype(jnp.float32)
            state, r, done = task.step(state, acts)
            r_np = np.array(r)
            fitness += np.where(done, 0, r_np)
            scored += (r_np > 0).astype(np.int32)
            conceded += (r_np < 0).astype(np.int32)
        return fitness, scored, conceded

    return evaluate


def _eval_selfplay_factory():
    """
    Returns eval_fn(genomes) → np.ndarray   (the signature NEAT expects).
    It compiles each genome once, calls selfplay_eval, and ignores global env.
    """
    def eval_fn(genomes):
        # 1. compile JAX forward fns
        params = [g.forward_jax() for g in genomes]
        policy = lambda p, obs, pst=None: p(obs)      # simple wrapper

        # 2. fresh RNG each generation
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        return np.asarray(selfplay_eval(params, policy, rng))
    return eval_fn

def train_slime(pop_size=20, n_generations=100, self_play=False, **kw):
    if self_play:
        eval_fn  = _eval_selfplay_factory()
        env_task = None 
    else:
        # … your original single-bot branch (leave untouched) …
        env_task = SlimeVolleyTask(pop_size, 1000)
        eval_fn  = _evaluate_factory(env_task)

    neat = NEAT(pop_size, GenomeTemplate(), eval_fn, env=env_task)
    neat.evolve(n_generations)
    return neat.best_genome

def _make_selfplay_eval(policy_fn, pop_size):
    """Returns eval_fn(list[Genome]) → np.ndarray."""
    def eval_fn(genomes):
        params   = [g.forward_jax() for g in genomes]
        rng      = jax.random.PRNGKey(np.random.randint(2**31))
        return np.asarray(selfplay_eval(params, policy_fn, rng))
    return eval_fn

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

__all__ = ["train_slime"]
