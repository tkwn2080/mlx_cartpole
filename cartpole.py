from typing import Tuple, Dict
import mlx.core as mx
import numpy as np
from enum import Enum, auto

# Define difficulty levels for Cartpole
class DifficultyLevel(Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    HARDEST = auto()

# General-purpose Cartpole environment
class MLXCartpole:
    def __init__(
        self, 
        batch_size: int, 
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        max_steps: int = 15000,
        g: float = 9.8,
        activity_threshold: float = 0.75
    ):
        """Initialize the cart-pole environment."""
        self.batch_size = batch_size
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.g = g
        self.activity_threshold = activity_threshold
        
        # State variables
        self.step_count = mx.zeros((batch_size,), dtype=mx.int32)
        self.state = mx.zeros((batch_size, 4))  # [x, x_dot, theta, theta_dot]
        self.done = mx.zeros((batch_size,), dtype=mx.bool_)
        self.do_nothing_count = mx.zeros((batch_size,), dtype=mx.int32)
        
        # Physical parameters
        self.length = 0.5
        self.masspole = 0.1
        self.total_mass = 1.1
        self.tau = 0.02
        self.max_theta = 12 * 2 * np.pi / 360  # 12 degrees in radians
        self.max_x = 2.4

        # Observation and action dimensions based on difficulty
        self.obs_dim = 2 if difficulty in [DifficultyLevel.HARD, DifficultyLevel.HARDEST] else 4
        self.action_dim = 3 if difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD] else 2
    
    def reset(self) -> mx.array:
        """Reset the environment to an initial state."""
        position = mx.random.uniform(-1.2, 1.2, (self.batch_size, 1))
        position_dot = mx.random.uniform(-0.05, 0.05, (self.batch_size, 1))
        angle = mx.random.uniform(-0.10475, 0.10475, (self.batch_size, 1))
        angle_dot = mx.random.uniform(-0.05, 0.05, (self.batch_size, 1))
        
        self.state = mx.concatenate([position, position_dot, angle, angle_dot], axis=1)
        self.step_count = mx.zeros((self.batch_size,), dtype=mx.int32)
        self.done = mx.zeros((self.batch_size,), dtype=mx.bool_)
        self.do_nothing_count = mx.zeros((self.batch_size,), dtype=mx.int32)
        
        if self.difficulty in [DifficultyLevel.HARD, DifficultyLevel.HARDEST]:
            return mx.stack([self.state[:, 0], self.state[:, 2]], axis=1)
        return self.state
    
    def get_observation(self) -> mx.array:
        """Get the current observation based on difficulty."""
        if self.difficulty in [DifficultyLevel.HARD, DifficultyLevel.HARDEST]:
            return mx.stack([self.state[:, 0], self.state[:, 2]], axis=1)
        return self.state
    
    def step(self, action: mx.array) -> Tuple[mx.array, mx.array, mx.array, Dict]:
        """Perform one step in the environment."""
        x, x_dot, theta, theta_dot = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]
        
        # Determine force based on action and difficulty
        if self.difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
            force = mx.zeros((self.batch_size,))
            force = mx.where(action == 0, -10.0, force)
            force = mx.where(action == 1, 10.0, force)
            if self.difficulty == DifficultyLevel.MEDIUM:
                self.do_nothing_count = self.do_nothing_count + mx.where(
                    (action == 2) & (~self.done), 1, 0
                )
        else:
            force = mx.where(action == 1, 10.0, -10.0)
        
        # Physics calculations
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.masspole * thetaacc * costheta / self.total_mass
        
        x_dot_new = x_dot + self.tau * xacc
        x_new = x + self.tau * x_dot_new
        theta_dot_new = theta_dot + self.tau * thetaacc
        theta_new = theta + self.tau * theta_dot_new
        
        self.state = mx.stack([x_new, x_dot_new, theta_new, theta_dot_new], axis=1)
        self.step_count = self.step_count + mx.where(self.done, 0, 1)
        
        # Check termination conditions
        terminated = (mx.abs(x_new) > self.max_x) | (mx.abs(theta_new) > self.max_theta)
        truncated = (self.step_count >= self.max_steps)
        done = terminated | truncated
        self.done = done
        
        # Calculate reward
        reward = mx.ones((self.batch_size,)) * 1.0
        if self.difficulty == DifficultyLevel.MEDIUM:
            do_nothing_pct = self.do_nothing_count / mx.maximum(self.step_count, mx.ones_like(self.step_count))
            reward = mx.where(
                do_nothing_pct < self.activity_threshold,
                self.do_nothing_count / self.activity_threshold,
                reward
            )
        reward = mx.where(done & terminated, 0.0, reward)
        
        next_state = self.get_observation()
        return next_state, reward, self.done, {}
