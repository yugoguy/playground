"""Training utilities for NEAT on Slime Volley."""

from typing import Callable, List
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from genome import Genome, InnovationTable
from neat import NEAT
from slimevolly import SlimeVolleyTask
from evojax.task.slimevolley import SlimeVolley

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

def train_slime(pop_size: int = 20,
                n_generations: int = 10,
                max_steps: int = 1000,
                self_play: bool = False,
                **kw):
    """
    If self_play=True the population learns by playing one another.
    Otherwise it plays the built-in reference AI.
    """
    # -----------------------------------------------------------
    # 1. get input/output sizes for the genome template
    probe_env = SlimeVolley(max_steps=max_steps, test=False)       # one-shot
    n_in, n_out = probe_env.obs_shape[0], probe_env.act_shape[0]
    template = Genome(n_in, n_out, InnovationTable())

    # -----------------------------------------------------------
    if self_play:
        # self-play evaluator
        policy = lambda p, obs, pst=None: p(obs)
        def eval_fn(genomes):
            params = [g.forward_jax() for g in genomes]
            rng    = jax.random.PRNGKey(np.random.randint(2**31))
            return np.asarray(selfplay_eval(params, policy, rng))
        env_task = None                        # evaluator makes its own env
    else:
        # single-bot path (original behaviour)
        env_task = SlimeVolley(max_steps=max_steps, test=True)
        def eval_fn(genomes):
            return env_task.rollout_batch([g.forward_jax() for g in genomes])

    # -----------------------------------------------------------
    neat = NEAT(pop_size, template, eval_fn)
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
@partial(jax.jit, static_argnums=(0, 1, 2))
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
    """
    pop_params : list of parameter pytrees (one per genome)
    policy_fn  : (params, obs, pst) -> action
    rng        : jax.random.PRNGKey
    returns    : np.ndarray of fitness values, shape (pop_size,)
    """
    pop_size = len(pop_params)
    perm     = np.random.permutation(pop_size)         # plain NumPy → scalars
    keys     = jax.random.split(rng, pop_size)

    fitness = []
    for i in range(pop_size):
        f = _play_match(policy_fn,
                        pop_params[i],
                        pop_params[perm[i]],           # scalar index is OK
                        keys[i])
        fitness.append(float(f))
    return np.asarray(fitness, dtype=np.float32)

__all__ = ["train_slime"]
