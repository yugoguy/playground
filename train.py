"""Training utilities for NEAT on Slime Volley."""

from typing import Callable, List
import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, InnovationTable
from neat import NEAT
from slimevolly import SlimeVolleyTask


def _evaluate_factory(task: SlimeVolleyTask) -> Callable[[List[Genome]], np.ndarray]:
    """Return an evaluation function for the given task."""

    pop_size = task.pop

    def evaluate(genomes: List[Genome]) -> np.ndarray:
        fwd_fns = [g.forward_jax() for g in genomes]
        state = task.reset()
        fitness = np.zeros(pop_size, dtype=np.float32)
        done = jnp.zeros(pop_size, dtype=bool)
        while not bool(done.all()):
            obs = jax.vmap(lambda s: s.obs)(state)
            logits = jnp.stack([f(obs[i]) for i, f in enumerate(fwd_fns)])
            acts = (
                (jnp.arange(3) & (jnp.argmax(logits, 1)[:, None] >> jnp.arange(3)))
                > 0
            ).astype(jnp.float32)
            state, r, done = task.step(state, acts)
            fitness += np.where(done, 0, np.array(r))
        return fitness

    return evaluate


def train_slime(
    pop_size: int = 20,
    n_generations: int = 10,
    max_steps: int = 1000,
) -> Genome:
    """Train a NEAT population on the SlimeVolley task.

    Returns the best genome after ``n_generations``.
    """

    task = SlimeVolleyTask(pop_size=pop_size, max_steps=max_steps)
    obs_dim = task.reset().obs.shape[-1]
    tbl = InnovationTable()
    template = Genome(obs_dim, 8, tbl)

    evaluate_fn = _evaluate_factory(task)
    neat = NEAT(pop_size=pop_size, genome_template=template, evaluate_fn=evaluate_fn)
    neat.evolve(n_generations)
    return neat.best_genome


__all__ = ["train_slime"]
