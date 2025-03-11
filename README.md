# MLX Cartpole with SNN Demo

This project leverages MLX, a high-performance machine learning framework optimized for Apple Silicon, to implement an accelerated version of the classic cart-pole problem. It serves three primary aims:

1. **MLX Acceleration**: Harnesses MLX's GPU acceleration on Apple Silicon for efficient computation of the cart-pole environment and spiking neural networks (SNNs).
2. **Neuromorphic Benchmark**: Implements the cart-pole problem as a benchmark for neuromorphic computing, inspired by "The Cart-Pole Application as a Benchmark for Neuromorphic Computing" by Plank et al. (2024).
3. **Neuroevolution Demo**: Demonstrates neuroevolution using the CMA-ES algorithm to train SNNs with the `SPIKE_FF_4` encoding type, showcasing a practical application of evolutionary optimization in neuromorphic systems.

## Project Structure

- **`cartpole.py`**: A general-purpose cart-pole environment (`MLXCartpole`) optimized with MLX for Apple Silicon.
- **`demo.py`**: A neuroevolution demo featuring a basic SNN implementation (`IzhikevichLayer`, `Network`), simulation logic, and experiment runner using `SPIKE_FF_4` encoding.

## Why MLX on Apple Silicon?

MLX provides seamless GPU acceleration on Apple Silicon (M1/M2/M3 chips), enabling fast matrix operations and neural network computations. This project uses MLX to accelerate the physics simulation in `MLXCartpole` and the SNN dynamics in `demo.py`, making it particularly efficient on macOS systems with Apple Silicon.

## Difficulty Levels

The cart-pole problem is implemented with four difficulty levels, as outlined in the neuromorphic benchmark paper:

- **EASY**: 4 observations (position, velocity, angle, angular velocity), 2 actions (left, right).
- **MEDIUM**: 4 observations, 3 actions (left, right, do-nothing) with an activity threshold.
- **HARD**: 2 observations (position, angle), 3 actions.
- **HARDEST**: 2 observations, 2 actions.
