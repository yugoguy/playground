# Minimal NEAT SlimeVolley Demo

This repository contains a very small implementation of the NEAT algorithm
adapted to work with [evojax](https://github.com/google/evojax)'s SlimeVolley
environment.  The code is intentionally short and is designed for educational
purposes.

The key modules are:

- `genome.py` – genome definition and mutation operators.
- `neat.py` – minimal NEAT evolution engine.
- `slimevolly.py` – light wrapper around `evojax.task.slimevolley.SlimeVolley`.
- `train.py` – utilities to train a population on SlimeVolley.
- `utils.py` – helpers for rendering rollouts to GIF files.

## Running in Google Colab

The following snippet shows how to train a small population and save a GIF of the
best agent.  It assumes that the Python files from this repository have been
uploaded to your Colab workspace.

```python
!pip install -q jax jaxlib evojax imageio

from train import train_slime
from utils import rollout_frames, save_gif

# Train for a few generations (increase for better performance)
best = train_slime(
    pop_size=20,
    n_generations=10,
    max_steps=1000,
    neat_kwargs={
        "mutate_weight_prob": 0.9,
        # add any other NEAT hyper-parameters here
    },
    seed=0,
    test=True,
    selfplay=True,
    pairing_fn=None,
)

# Roll the trained genome and save a gif
frames = rollout_frames(best, steps=400, frame_skip=5)
save_gif(frames, "slime.gif")
```

The optional ``neat_kwargs`` argument allows you to tweak NEAT
hyper-parameters such as mutation rates.  Setting ``selfplay=True`` enables
training agents against each other instead of the built-in opponent.

### Self-play training algorithm

When self-play is enabled the population is divided into pairs and every pair
plays a match in parallel.  Fitness is simply the cumulative reward from each
game.  The pairing strategy is controlled by ``pairing_fn`` so more elaborate
schedules (e.g. round‑robin) can be plugged in later.  This setup provides a
denser learning signal compared with the default fixed opponent.

This will produce a file `slime.gif` which can be displayed directly in Colab
or downloaded.

## License

This project is provided for demonstration purposes only.  See the repository
for more details.
