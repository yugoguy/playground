# Neuroevolution Playground

This repository contains a minimal implementation of NEAT (Neuroevolution of 
Augmenting Topologies) and helper utilities to evolve agents for
Neural Slime Volleyball.  The code is designed to be easily imported in a
Google Colab notebook.

## Training

Use `train.train_slime` to evolve a population of genomes.  It returns the
best genome after a number of generations.

```python
from train import train_slime
best_genome = train_slime(pop_size=10, n_generations=3)
```

## Visualisation

The utilities in `utils.py` can roll out a genome in the environment and
produce a GIF.

```python
from utils import rollout_frames, save_gif
frames = rollout_frames(best_genome)
save_gif(frames, "neat.gif")
```

<<<<<<< ours
=======
## Example (Google Colab)

Below is a minimal example that trains a population and visualises the best
agent.  Assuming the repository files have been uploaded to Colab, run:

```python
from train import train_slime
from utils import rollout_frames, save_gif
from IPython.display import Image, display

# evolve a small population for a few generations
best = train_slime(pop_size=10, n_generations=3)

# create a GIF of the trained agent
frames = rollout_frames(best)
save_gif(frames, "neat_best.gif")
display(Image("neat_best.gif"))
```

>>>>>>> theirs
---
