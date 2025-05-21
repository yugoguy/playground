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
        return np.asarray(selfplay_eval(params, policy))
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
    n_in  = probe_env.obs_shape[-1]  
    n_out = probe_env.act_shape[-1]   
    template = Genome(n_in, n_out, InnovationTable())

    # -----------------------------------------------------------
    if self_play:
        # self-play evaluator
        policy = lambda p, obs, pst=None: p(obs)
        def eval_fn(genomes):
            params = [g.forward_jax() for g in genomes]
            return np.asarray(selfplay_eval(params, policy))
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

# ─────────────────────────────────────────────────────────────
def _play_match(policy_fn, params_a, params_b):
    """
    One 1 000-step SlimeVolley game, batched API handled properly.
    """
    key   = jax.random.PRNGKey(np.random.randint(2**31))
    key   = key[None, :]                           # (1,2)
    env   = SlimeVolley(max_steps=1000, test=False)
    state = env.reset(key)                         # state.obs shape (1, 2, obs_dim)

    pst_a = pst_b = None
    total = 0.0
    for _ in range(env.max_steps):
        # state.obs[0] → (2, obs_dim).  Index player 0 and 1 explicitly.
        obs_pair = state.obs[0]
        obs_a, obs_b = obs_pair[0], obs_pair[1]

        act_a, pst_a = policy_fn(params_a, obs_a, pst_a)
        act_b, pst_b = policy_fn(params_b, obs_b, pst_b)

        # (1,2,3) batched action tensor
        acts = jnp.stack([act_a, act_b])[None, :]

        state, r, done = env.step(state, acts)     # r shape (1,)
        total += 10.0 * float(r[0]) + 0.01         # shaped reward
        if bool(done[0]):
            break

    return total


def selfplay_eval(pop_params, policy_fn):
    """
    Each genome plays ONE match vs. a random opponent.
    Returns np.ndarray of float32 fitness values.
    """
    pop_size = len(pop_params)
    perm     = np.random.permutation(pop_size)           # Python ints
    fitness  = np.empty(pop_size, dtype=np.float32)

    for i in range(pop_size):
        fitness[i] = _play_match(policy_fn,
                                 pop_params[i],
                                 pop_params[perm[i]])
    return fitness


__all__ = ["train_slime"]
