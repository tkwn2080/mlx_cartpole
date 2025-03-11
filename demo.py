import matplotlib
matplotlib.use('Agg')
import mlx.core as mx
import numpy as np
from cartpole import DifficultyLevel, MLXCartpole
import cma
import os
import shutil
import datetime
import csv
import matplotlib.pyplot as plt
import time
from visualise import generate_video  # Import the visualization function

# Izhikevich Neuron Layer
class IzhikevichLayer:
    def __init__(self, population_size, n_neurons, n_trials, a=0.02, b=0.2, c=-65, d=6, dt=1.0):
        """Initialize an Izhikevich neuron layer."""
        self.population_size = population_size
        self.n_trials = n_trials
        self.n_neurons = n_neurons
        self.a = mx.full([population_size, n_neurons], a) if isinstance(a, (int, float)) else mx.array(a)
        self.b = mx.full([population_size, n_neurons], b) if isinstance(b, (int, float)) else mx.array(b)
        self.c = mx.full([population_size, n_neurons], c) if isinstance(c, (int, float)) else mx.array(c)
        self.d = mx.full([population_size, n_neurons], d) if isinstance(d, (int, float)) else mx.array(d)
        self.dt = dt
        v_init = -65
        self.v = mx.full([self.population_size, self.n_trials, self.n_neurons], v_init)
        self.u = mx.broadcast_to(self.b[:, None, :] * v_init, [self.population_size, self.n_trials, self.n_neurons])

    def reset(self, v_init=-65):
        """Reset neuron potentials."""
        self.v = mx.full([self.population_size, self.n_trials, self.n_neurons], v_init)
        self.u = mx.broadcast_to(self.b[:, None, :] * v_init, [self.population_size, self.n_trials, self.n_neurons])

    def update(self, I):
        """Update neuron states and return spikes."""
        dv = self.dt * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        du = self.dt * self.a[:, None, :] * (self.b[:, None, :] * self.v - self.u)
        self.v += dv
        self.u += du
        spikes = (self.v >= 30).astype(mx.float32)
        self.v = mx.where(spikes, self.c[:, None, :], self.v)
        self.u = mx.where(spikes, self.u + self.d[:, None, :], self.u)
        return spikes

# Spiking Neural Network with SPIKE_FF_2 Encoding
class Network:
    def __init__(self, population_size, n_trials, n_input, n_hidden, n_output, timesteps, dt=1.0):
        """Initialize the SNN with input, hidden, and output layers."""
        self.population_size = population_size
        self.n_trials = n_trials
        self.timesteps = timesteps
        
        # Split hidden layer into excitatory and inhibitory neurons
        self.n_hidden_ex = n_hidden * 4 // 5
        self.n_hidden_in = n_hidden // 5
        assert self.n_hidden_ex + self.n_hidden_in == n_hidden, "Hidden neuron split mismatch"

        # Input layer
        self.input_layer = IzhikevichLayer(self.population_size, n_input, n_trials, a=0.02, b=0.2, c=-65, d=8, dt=dt)
        
        # Hidden layer with excitatory and inhibitory neurons
        hidden_a = mx.concatenate([
            mx.full((self.population_size, self.n_hidden_ex), 0.02),
            mx.full((self.population_size, self.n_hidden_in), 0.1)
        ], axis=1)
        hidden_b = mx.concatenate([
            mx.full((self.population_size, self.n_hidden_ex), 0.2),
            mx.full((self.population_size, self.n_hidden_in), 0.25)
        ], axis=1)
        hidden_c = mx.full((self.population_size, n_hidden), -65)
        hidden_d = mx.concatenate([
            mx.full((self.population_size, self.n_hidden_ex), 8),
            mx.full((self.population_size, self.n_hidden_in), 2)
        ], axis=1)
        self.hidden_layer = IzhikevichLayer(self.population_size, n_hidden, n_trials, 
                                            a=hidden_a, b=hidden_b, c=hidden_c, d=hidden_d, dt=dt)
        
        # Output layer
        self.output_layer = IzhikevichLayer(self.population_size, n_output, n_trials, 
                                            a=0.02, b=0.2, c=-65, d=8, dt=dt)

        # Initialize weights
        self.weights_input_hidden = mx.random.uniform(0, 5, [self.population_size, n_input, n_hidden])
        w_ho_ex = mx.random.uniform(0, 5, [self.population_size, self.n_hidden_ex, n_output])
        w_ho_in = mx.random.uniform(-2.5, 0, [self.population_size, self.n_hidden_in, n_output])
        self.weights_hidden_output = mx.concatenate([w_ho_ex, w_ho_in], axis=1)

    def reset(self):
        """Reset all layers."""
        self.input_layer.reset()
        self.hidden_layer.reset()
        self.output_layer.reset()

    def set_parameters(self, genotype):
        """Set network weights from genotype."""
        n_in = self.input_layer.n_neurons
        n_hid = self.hidden_layer.n_neurons
        n_out = self.output_layer.n_neurons
        n_hid_ex = self.n_hidden_ex
        
        n_ih = n_in * n_hid
        self.weights_input_hidden = 5 * mx.sigmoid(genotype[:, :n_ih]).reshape([self.population_size, n_in, n_hid])
        weights_ho_start = n_ih
        n_ho = n_hid * n_out
        w_ho_flat = genotype[:, weights_ho_start:weights_ho_start + n_ho].reshape([self.population_size, n_hid, n_out])
        self.weights_hidden_output = mx.concatenate([
            5 * mx.sigmoid(w_ho_flat[:, :n_hid_ex, :]),
            -2.5 * mx.sigmoid(w_ho_flat[:, n_hid_ex:, :])
        ], axis=1)

    def call(self, raw_inputs):
        """Process inputs through the network and return output spikes."""
        self.reset()
        
        max_vals = mx.array([2.4, 5.0, 0.2095, 5.0])  # Termination bounds
        max_spikes = 20.0  # Increased for better sensitivity
        n_obs = raw_inputs.shape[-1] // 2  # Number of observations
        t = mx.arange(self.timesteps)

        # SPIKE_FF_2 encoding: 2 bins (neg, pos) per observation
        spikes_input = mx.zeros((self.population_size, self.n_trials, self.timesteps, 2 * n_obs))
        abs_inputs = mx.abs(raw_inputs)  # Shape: (pop_size, n_trials, 2 * n_obs)
        max_vals_ff2 = mx.concatenate([max_vals[:n_obs], max_vals[:n_obs]], axis=0)  # Shape: (2 * n_obs,)
        
        spike_counts = mx.minimum((abs_inputs / max_vals_ff2[None, None, :]) * max_spikes, max_spikes)
        spike_counts = mx.round(spike_counts).astype(mx.int32)  # Shape: (pop_size, n_trials, 2 * n_obs)
        
        interval = mx.where(spike_counts > 0, self.timesteps // spike_counts, self.timesteps)
        spike_condition = (t[:, None, None, None] % interval[None, :, :, :] == 0) & \
                         (t[:, None, None, None] < spike_counts[None, :, :, :] * interval[None, :, :, :])
        spikes_input = mx.where(spike_condition.transpose((1, 2, 0, 3)), 1.0, 0.0)

        # Process through hidden and output layers
        output_spikes = mx.zeros([self.population_size, self.n_trials, self.output_layer.n_neurons])
        for t in range(self.timesteps):
            I_hidden = mx.matmul(spikes_input[:, :, t, :], self.weights_input_hidden)
            spikes_hidden = self.hidden_layer.update(I_hidden)
            I_output = mx.matmul(spikes_hidden, self.weights_hidden_output)
            spikes_output = self.output_layer.update(I_output)
            output_spikes += spikes_output
        
        return output_spikes

def run_simulation(net, env, max_steps=1000):
    """Run the simulation with the network and environment."""
    population_size = net.population_size
    n_trials = net.n_trials
    batch_size = population_size * n_trials
    assert env.batch_size == batch_size
    
    obs = env.reset()
    done = mx.zeros((batch_size,), dtype=mx.bool_)
    
    for step in range(max_steps):
        if mx.all(done):
            break
        
        obs_reshaped = obs.reshape(population_size, n_trials, -1)
        pos_parts = mx.maximum(obs_reshaped, 0)
        neg_parts = mx.maximum(-obs_reshaped, 0)
        raw_inputs = mx.concatenate([pos_parts, neg_parts], axis=-1)
        
        output_spikes = net.call(raw_inputs)
        actions = mx.argmax(output_spikes, axis=-1)  # Vote decoding
        actions_flat = actions.reshape(batch_size)
        
        obs, reward, done, _ = env.step(actions_flat)
        
        mx.eval(obs, reward, done)
    
    step_counts = env.step_count.reshape(population_size, n_trials)
    if env.difficulty == DifficultyLevel.MEDIUM:
        do_nothing_counts = env.do_nothing_count.reshape(population_size, n_trials)
        ratio = do_nothing_counts / step_counts
        fitness_per_trial = mx.where(
            ratio > env.activity_threshold,
            step_counts,
            do_nothing_counts / env.activity_threshold
        )
        fitness = mx.mean(fitness_per_trial, axis=1)
    else:
        fitness = mx.mean(step_counts, axis=1)
    
    return fitness

def run_visualisation_trial(net, env, genotype, max_steps=1000):
    """Run a single trial with the given genotype and record states and actions."""
    net.set_parameters(genotype[None, :])  # Set parameters for one individual
    obs = env.reset()
    done = False
    states = []
    actions_list = []
    
    for step in range(max_steps):
        if done:
            break
        
        obs_reshaped = obs.reshape(1, 1, -1)  # population_size=1, n_trials=1
        pos_parts = mx.maximum(obs_reshaped, 0)
        neg_parts = mx.maximum(-obs_reshaped, 0)
        raw_inputs = mx.concatenate([pos_parts, neg_parts], axis=-1)
        
        output_spikes = net.call(raw_inputs)
        action = mx.argmax(output_spikes, axis=-1).item()  # Single action
        
        states.append(env.state[0].tolist())  # Record state
        actions_list.append(action)
        
        obs, reward, done, _ = env.step(mx.array([action]))
        mx.eval(obs, reward, done)
    
    return states, actions_list

def main(test_mode=False, difficulty: DifficultyLevel = DifficultyLevel.EASY):
    """Run the SNN experiment with SPIKE_FF_2 encoding."""
    if test_mode:
        print("Starting test run...")
        population_size = 20
        n_generations = 20
        n_trials = 20
        max_steps = 500
        test_dir = "output/test"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)
        output_dir = test_dir
    else:
        print("Starting full experiment...")
        population_size = 500
        n_generations = 200
        n_trials = 50
        max_steps = 15000
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)

    # Set observation and action dimensions based on difficulty
    if difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]:
        n_obs = 4
    else:  # HARD or HARDEST
        n_obs = 2
    n_input = n_obs * 2  # SPIKE_FF_2: 2 bins per observation
    n_output = 3 if difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD] else 2
    n_hidden = 20
    timesteps = 20

    max_fitness = []

    print(f"\nRunning SPIKE_FF_2 Experiment with difficulty {difficulty.name}")
    net = Network(population_size, n_trials, n_input, n_hidden, n_output, timesteps)
    batch_size = population_size * n_trials
    env = MLXCartpole(batch_size=batch_size, difficulty=difficulty, max_steps=max_steps)

    genotype_length = n_input * n_hidden + n_hidden * n_output
    es = cma.CMAEvolutionStrategy([0.0] * genotype_length, 1.0, {'popsize': population_size})

    csv_file = os.path.join(output_dir, f"SPIKE_FF_2_{difficulty.name}_stats.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean", "std", "min", "max"])

        for generation in range(n_generations):
            start_time = time.time()
            
            solutions = es.ask()
            genotypes = mx.array(np.stack(solutions, axis=0))
            net.set_parameters(genotypes)
            fitness = run_simulation(net, env, max_steps=max_steps)
            es.tell(solutions, (-fitness).tolist())

            mean_fitness = float(mx.mean(fitness))
            std_fitness = float(mx.std(fitness))
            min_fitness = float(mx.min(fitness))
            max_fitness_val = float(mx.max(fitness))
            writer.writerow([generation + 1, mean_fitness, std_fitness, min_fitness, max_fitness_val])
            max_fitness.append(max_fitness_val)
            
            generation_time = time.time() - start_time
            print(f"Generation {generation+1}, Best fitness: {max_fitness_val:.1f}, Time: {generation_time:.1f}s")

    # Save top individuals
    num_top = min(5, population_size)
    top_indices = mx.argpartition(-fitness, kth=num_top-1)[:num_top]
    top_fitness = fitness[top_indices].tolist()
    print(f"Top {num_top} fitness values: {top_fitness}")

    # Visualize the best performer
    best_genotype = es.result.xbest
    net_vis = Network(1, 1, n_input, n_hidden, n_output, timesteps)  # Single individual, single trial
    env_vis = MLXCartpole(batch_size=1, difficulty=difficulty, max_steps=max_steps)
    n_visualise_trials = 3
    print(f"Generating visualizations for {n_visualise_trials} trials of the best performer...")
    for trial in range(n_visualise_trials):
        states, actions = run_visualisation_trial(net_vis, env_vis, best_genotype, max_steps=max_steps)
        generate_video(states, actions, output_dir, trial, difficulty)
        print(f"Generated video for trial {trial + 1}: {os.path.join(output_dir, f'trial_{trial}.mp4')}")

    # Plot performance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_generations + 1), max_fitness, label=f"SPIKE_FF_2 - {difficulty.name}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Steps Survived)")
    plt.title(f"SPIKE_FF_2 Performance on {difficulty.name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"SPIKE_FF_2_{difficulty.name}_performance.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SPIKE_FF_2 experiment')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    if args.test:
        main(test_mode=True, difficulty=DifficultyLevel.EASY)
        print("SPIKE_FF_2 test completed successfully.")
    else:
        main(test_mode=False, difficulty=DifficultyLevel.EASY)
        print("SPIKE_FF_2 full experiment completed successfully.")
