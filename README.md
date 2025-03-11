# MLX Cartpole with SNN Experiment

This is a personal project I put together to explore spiking neural networks (SNNs) and reinforcement learning on my Mac. It's a lightweight implementation of the classic cart-pole balancing problem, optimized to run efficiently on Apple Silicon (M1/M2/M3) using MLX. If you're interested in neuromorphic computing or neuroevolution, this might be a fun starting point for your own experiments.

## What It Does
* **SNNs with Apple Silicon acceleration**: Uses MLX to speed up physics and neural network simulations, making it fast enough to run locally on a MacBook or Mac Mini.
* **Neuroevolution demo**: Trains spiking networks using CMA-ES (a genetic algorithm) instead of traditional backpropagation.
* **Four difficulty levels**: Ranging from "easy" (full observations) to "hardest" (limited inputs), so you can test how well your SNN handles different challenges.

## Why I Built It
I wanted to see if I could train a spiking neural network locally without relying on cloud GPUs. Apple Silicon's unified memory architecture and GPU make it a great platform for small-scale experiments like this—it's fast, quiet, and doesn't overheat. The cart-pole problem is a simple but effective benchmark for testing SNNs, and I thought it'd be interesting to see how well they perform with evolutionary training.

## How It Works
* `cartpole.py`: Handles the physics simulation, accelerated with MLX.
* `demo.py`: Runs the neuroevolution loop, using Izhikevich neurons for the SNN (a good balance of simplicity and bio-realism).

## Why Apple Silicon?
Because it's what I have, and it's surprisingly capable for this kind of work. The way Apple Silicon handles memory and graphics together means MLX can run experiments efficiently without needing high-end hardware. It's a great way to experiment with AI concepts on consumer-grade devices.

## For Others Who Might Be Interested
If you're into:
* **Neuromorphic computing** (SNNs that use spikes instead of traditional neurons),
* **Evolutionary algorithms** (training networks without gradients),
* **Local experimentation** (no cloud dependencies),

...this project might be a useful reference or starting point. It's not polished or production-ready, but it's been a fun way for me to learn and experiment.
