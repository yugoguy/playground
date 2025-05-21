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
<<<<<<< ours
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
=======
produce a GIF.  Below is a full snippet you can run in Google Colab to train
an agent and visualise the result:

```python
from pyvirtualdisplay import Display
from IPython.display import display, Image
from utils import rollout_frames, save_gif

# start a virtual display (needed in Colab)
Display(visible=0, size=(1400, 900)).start()

best_genome = train_slime(pop_size=10, n_generations=3)
frames = rollout_frames(best_genome)
>>>>>>> theirs
save_gif(frames, "neat_best.gif")
display(Image("neat_best.gif"))
```

<<<<<<< ours
>>>>>>> theirs
---
=======
>>>>>>> theirs
