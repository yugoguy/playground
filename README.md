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
best = train_slime(pop_size=20, n_generations=10, max_steps=1000)

# Roll the trained genome and save a gif
frames = rollout_frames(best, steps=400, frame_skip=5)
save_gif(frames, 'slime.gif')
```

This will produce a file `slime.gif` which can be displayed directly in Colab
or downloaded.

## License

This project is provided for demonstration purposes only.  See the repository
for more details.
