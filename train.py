"""Training utilities for NEAT on Slime Volley."""

from typing import Callable, List
import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, InnovationTable
from neat import NEAT
from slimevolly import SlimeVolleyTask
from utils import selfplay_eval


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


def _selfplay_evaluate_factory(
    task: SlimeVolleySelfPlayTask,
    pairing_fn: Callable[[int], List[tuple[int, int]]] | None = None,
) -> Callable[[List[Genome]], np.ndarray]:
    """Return an evaluation function for self-play matches."""

    n_pairs = task.n_pairs
    if pairing_fn is None:
        def pairing_fn(n: int) -> List[tuple[int, int]]:
            return [(2 * i, 2 * i + 1) for i in range(n // 2)]

    def evaluate(genomes: List[Genome]) -> np.ndarray:
        assert len(genomes) == n_pairs * 2, "population must be even"
        pairs = pairing_fn(len(genomes))
        fwd = [g.forward_jax() for g in genomes]
        state = task.reset()
        fitness = np.zeros(len(genomes), dtype=np.float32)
        done = jnp.zeros(n_pairs, dtype=bool)
        while not bool(done.all()):
            obs1 = state.obs[:, 0]
            obs2 = state.obs[:, 1]
            logits_a = jnp.stack([fwd[p[0]](obs1[i]) for i, p in enumerate(pairs)])
            logits_b = jnp.stack([fwd[p[1]](obs2[i]) for i, p in enumerate(pairs)])
            state, r, done = task.step(state, logits_a, logits_b)
            r_np = np.array(r)
            for i, (p0, p1) in enumerate(pairs):
                fitness[p0] += r_np[i, 0]
                fitness[p1] += r_np[i, 1]
        return fitness

    return evaluate


def train_slime(pop_size=20,
                n_generations=100,
                self_play=False,
                eval_fn=None,
                **kwargs):

    # pick evaluation function -------------------------------------
    if self_play:
        policy_fn = forward_fn_for_genome
        eval_fn   = eval_fn or (lambda p, rng:
                                selfplay_eval(p, policy_fn, rng))
        env = None
    else:
        from slimevolly import SlimeVolley
        env     = SlimeVolley(max_steps=1000, test=True)
        eval_fn = eval_fn or default_single_eval

    neat = NEAT(pop_size=pop_size,
                template=GenomeTemplate(),
                eval_fn=eval_fn,
                env=env)
    neat.run(n_generations)
    return neat.best_genome


__all__ = ["train_slime"]
