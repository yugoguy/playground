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


def train_slime(
    pop_size: int = 20,
    n_generations: int = 10,
    max_steps: int = 1000,
    neat_kwargs: dict | None = None,
) -> Genome:
    """Train a NEAT population on the SlimeVolley task.

    Additional NEAT hyper-parameters can be provided via ``neat_kwargs``.

    Returns the best genome after ``n_generations``.
    """

    task = SlimeVolleyTask(pop_size=pop_size, max_steps=max_steps)
    obs_dim = task.reset().obs.shape[-1]
    tbl = InnovationTable()
    template = Genome(obs_dim, 8, tbl)

    evaluate_fn = _evaluate_factory(task)
    if neat_kwargs is None:
        neat_kwargs = {}
    neat = NEAT(pop_size=pop_size,
                genome_template=template,
                evaluate_fn=evaluate_fn,
                **neat_kwargs)
    neat.evolve(n_generations)
    return neat.best_genome


__all__ = ["train_slime"]
