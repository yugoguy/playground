<<<<<<< HEAD
<<<<<<< ours
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
=======
Part 1: Neuroevolution

Train a NEAT network to play Neural Slime Volleyball
The purpose of this section is to ask you to explore an old neuroevolution algorithm, called
Neuroevolution of augmenting topologies (NEAT), which can be used to evolve neural
network architectures and their weights at the same time. Please study how NEAT works, by
studying the materials in [1] and [2] in the References section of this section. The purpose of
this part is also to challenge one to think from first principles, in the era of boilerplate
frameworks.

NEAT -like methods have been used in more recent papers [3], to demonstrate that very
minimal neural network structures, with various activation functions, can be evolved to
perform various tasks, and such networks have great inductive biases for the tasks they have
been evolved for, even if the weight parameters don't even need to be trained. Some examples
of network architectures produced by NEAT -like algorithm in [3]:
<img width="784" alt="スクリーンショット 2025-05-21 14 28 16" src="https://github.com/user-attachments/assets/94b6534b-9173-4c55-982f-c3cebeb9041b" />

We want you to implement NEAT in NumPy or JAX (feel free to use and reference code from
resources in [1]), and your task is to evolve a NEAT agent to beat the internal Al in Neural
Slime Volleyball. You can access the environment in [4] for NumPy or [5] for JAX. Note: We
want your NEAT networks to be strictly feed-forward networks. No recurrent connections.
<img width="406" alt="スクリーンショット 2025-05-21 14 28 29" src="https://github.com/user-attachments/assets/a0bb4873-0811-4d2d-aebb-bb132accef7d" />

There are two ways to evolve NEAT ̶ either to evolve the network to directly play (and
eventually beat) the internal agent (which uses a simple fixed, fully-connected neural
network), or to evolve NEAT networks to play against each other via self-play, where
eventually, the best network might beat the internal built-in agent. It is up to you how you
evolve your networks.

Please document what you did: produce visualizations of the types of networks your algorithm
produced. Include some GIF animations of your NEAT agent playing against the internal one
and put the animation in your Google Doc report We would like you to write and comment
on how the networks complexified, and whether there are interesting aspects you see worth
writing about ̶ such as the types of activation functions it ends up using, or particular
structures you find interesting. The EvoJAX or Gym Env code in [4] and [5] might be a bit
dated, from a few years ago, so you may need to make some modifications to get it to work.

Implement Backprop NEAT: The Grandfather of Neural Architecture Search
One of the great things about JAX is that it looks very similar to NumPy, but makes gradients
very easy to calculate. Actually, an old blog post [2] attempted to use NEAT to evolve the
architecture, but use the backpropagation algorithm to solve for the weights of the
architecture, so that a minimal network can be evolved and trained to perform a simple
classification task.

Example of NEAT evolving networks to solve simple classification tasks. Backprop is used to
train the weights, while NEAT is only used for evolving the architectures. The inputs are x
and y-axis, and output is a logistic regression classifier for a 2-class classification problem.
We want you to modify your implementation to work in JAX, and limit your activation
functions to be differentiable ones (including mostly differentiable ones like ReLU).
Implement Backprop NEAT, and demonstrate / visualize the best evolved networks for the
toy classification 2D tasks in [2]. If you come from a strong engineering background, see what
you can do to improve the performance of the NEAT + gradient combination for your
Backprop NEAT implementation.

There is a GitHub repo in [2] that contains simple JS code to generate dataset for
classification task like Circle, Spiral, XOR ̶ make sure you can evolve feed-forward networks
architectures (where the weights are trained with backprop via JAX) to classify those patterns.
Put in some penalty for architecture complexity (to encourage simple and elegant
architectures), and in your report, visualize various architectures that you find interesting.

References
[1] NEAT resources Original Paper: Evolving Neural Networks Through Augmenting
Topologies (2002) httos://nn.cs.utexas.edu/?stanley:ec02 VVikipedia https.//An
wikiperiia nrg/wiki/NAtirnevollition_of Aiigmenting_topologieR PrettyNEAT (python
version we did at Google Brain for WANN experiments) MtpsVgithilh corn/googIA/hrain-
tokyo-wnrkshnp/tme/masterNVANNRelPase/prettyNFAT neat-python
httos://github.com/CodeReclaimers/neat-oython Minki Oka/Riftg's recent book
(Japanese) published by O'Reilly Japan Python-011U (Y,A--7:11.11-:std:A4LrItJ7)1,19 XL3,
(I think it uses neat-python) https-//www co.jp/hooks/9784814400003/ Mig1 NEAT -
PythononL (OJNI4Mgib httos://oreilIv-janan.githubio/OpenEndedCodebook/ann1/

[2] Backprop NEAT Blog post https.//hIng.otorn net/2016/05/07/hankprnp-neat/ Github
(for generating Two Circle, XOR, and Spiral datasets)
httos://github.com/hardmaru/backproo-neat-is/

[3] Weight Agnostic Neural Network (see open-source code repo for NEAT examples) https-
//weightagnnstin githith in/

[4] Neural Slime Volleyball Gym Environment https-//githiihcom/hardmandslimpvolleygym

[5] EvoJAX's port of Neural Slime Volleyball environment httos://github.com/google/evoiax

## Quick Start (Google Colab)

Below is a minimal example that trains a small population and saves a GIF of the best agent.
The helper functions are provided in `train.py` and `utils.py`.

```python
from pyvirtualdisplay import Display
Display(visible=0, size=(1400, 900)).start()

from train import train_slime
from utils import rollout_frames, save_gif

best_genome = train_slime(pop_size=10, n_generations=3)
frames = rollout_frames(best_genome)
save_gif(frames, "neat.gif")
```
>>>>>>> theirs
=======

>>>>>>> f2d1e5d03bc1b77c38a8b6f502e4ed38a0b24400
