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
        """NEAT が呼び出す評価関数。selfplay_eval を 1 回呼ぶだけ。"""
        params = [g.forward_jax() for g in genomes]          # list[PyTree]
    
        # policy_fn は (params, obs) -> action だけを返すシンプル版
        policy_fn = lambda net_params, obs: net_params(obs)
    
        return np.asarray(selfplay_eval(params, policy_fn))  # shape (pop_size,)
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
            """NEAT が呼び出す評価関数。selfplay_eval を 1 回呼ぶだけ。"""
            params = [g.forward_jax() for g in genomes]          # list[PyTree]
        
            # policy_fn は (params, obs) -> action だけを返すシンプル版
            policy_fn = lambda net_params, obs: net_params(obs)
        
            return np.asarray(selfplay_eval(params, policy_fn))  # shape (pop_size,)
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
    key   = jax.random.PRNGKey(np.random.randint(2**31))[None, :]
    env   = SlimeVolley(max_steps=1000, test=False)
    state = env.reset(key)

    total = 0.0
    half  = state.obs.shape[-1] // 2            # 12

    for _ in range(env.max_steps):
        obs_vec = state.obs[0]                  # (24,)
        obs_a   = obs_vec[:half]                # right player's 12 features

        act_a = policy_fn(params_a, obs_a)      # (3,)
        acts  = jnp.expand_dims(act_a, 0)       # (1,3)

        state, r, done = env.step(state, acts)
        total += 10.0 * float(r[0]) + 0.01
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
