# MLX Cartpole

This project brings the classic cart-pole problem to your Apple Silicon device (M1/M2/M3 Macs) using MLX, a high-performance machine learning framework optimized for Apple's GPU acceleration. Designed for local experimentation, it enables homebrew enthusiasts to explore reinforcement learning (RL), neuroevolution, and neuromorphic computing right from their MacBooks or Mac Minis. With a lightweight setup and efficient computation, it’s perfect for tinkering with cutting-edge AI concepts on consumer hardware.

## Aims

1. **MLX Acceleration on Apple Silicon**: Leverages MLX to accelerate cart-pole physics and spiking neural networks (SNNs) on Apple Silicon GPUs, making it fast and efficient for local experimentation.
2. **Neuromorphic Benchmark**: Implements the cart-pole problem as a benchmark for neuromorphic computing, based on "The Cart-Pole Application as a Benchmark for Neuromorphic Computing" by Plank et al. (2024), offering a standardized playground for SNN research.
3. **Neuroevolution Demo**: Demonstrates neuroevolution with the CMA-ES algorithm to train SNNs using the `SPIKE_FF_2` encoding type, providing a hands-on example of evolutionary optimization in neuromorphic systems.

## Why Apple Silicon?

Apple Silicon’s unified memory architecture and powerful GPU make it an ideal platform for local AI experimentation. MLX taps into this hardware, accelerating matrix operations and neural network dynamics without requiring high-end servers or cloud resources. Whether you’re a hobbyist with a MacBook Air or a researcher with a Mac Studio, this project lets you run sophisticated RL and neuromorphic experiments at home.

## Project Structure

- **`cartpole.py`**: A general-purpose, MLX-accelerated cart-pole environment (`MLXCartpole`) for flexible experimentation.
- **`demo.py`**: A neuroevolution demo with a full SNN implementation (`IzhikevichLayer`, `Network`), simulation logic, and experiment runner using `SPIKE_FF_2` encoding.
- **`README.md`**: This guide.

## Difficulty Levels

Experiment with four difficulty levels inspired by the neuromorphic benchmark paper:

- **EASY**: 4 observations (position, velocity, angle, angular velocity), 2 actions (left, right).
- **MEDIUM**: 4 observations, 3 actions (left, right, do-nothing) with an activity threshold.
- **HARD**: 2 observations (position, angle), 3 actions.
- **HARDEST**: 2 observations, 2 actions.

These levels let you tweak the challenge and test your SNN designs locally.
