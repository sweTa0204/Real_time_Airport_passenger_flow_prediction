"""
Reinforcement Learning module for Staff Scheduling Optimization
Uses Stable-Baselines3 with a custom Gym environment
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import os


class AirportStaffSchedulingEnv(gym.Env):
    """
    Custom Gym environment for optimizing airport staff allocation
    
    State: Current passenger flow, queue lengths, time of day, current staff allocation
    Actions: Adjust staff allocation across different zones
    Reward: Minimize wait times while controlling labor costs
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or {
            'max_steps': 96,  # 24 hours at 15-min intervals
            'zones': ['security', 'checkin', 'baggage'],
            'max_staff_per_zone': 20,
            'min_staff_per_zone': 2,
            'staff_cost_per_hour': 25,
            'wait_time_penalty': 10,
            'target_wait_times': {'security': 15, 'checkin': 8, 'baggage': 12}
        }
        
        self.n_zones = len(self.config['zones'])
        self.max_staff = self.config['max_staff_per_zone']
        self.min_staff = self.config['min_staff_per_zone']
        
        # Action space: Change in staff for each zone (-2, -1, 0, +1, +2)
        self.action_space = spaces.MultiDiscrete([5] * self.n_zones)
        
        # Observation space: [hour, dow, passenger_flow, queue_lengths(3), wait_times(3), staff_allocation(3)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0]*3 + [0]*3 + [self.min_staff]*3),
            high=np.array([23, 6, 1000] + [200]*3 + [60]*3 + [self.max_staff]*3),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.hour = 6  # Start at 6 AM
        self.day_of_week = np.random.randint(0, 7)
        
        # Initialize staff allocation
        self.staff_allocation = {
            'security': 8,
            'checkin': 10,
            'baggage': 5
        }
        
        # Initialize metrics
        self.queue_lengths = {'security': 20, 'checkin': 15, 'baggage': 10}
        self.wait_times = {'security': 10, 'checkin': 5, 'baggage': 8}
        
        self.total_reward = 0
        self.episode_stats = {'wait_times': [], 'staff_costs': [], 'rewards': []}
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        passenger_flow = self._get_passenger_flow(self.hour, self.day_of_week)
        
        obs = np.array([
            self.hour,
            self.day_of_week,
            passenger_flow,
            self.queue_lengths['security'],
            self.queue_lengths['checkin'],
            self.queue_lengths['baggage'],
            self.wait_times['security'],
            self.wait_times['checkin'],
            self.wait_times['baggage'],
            self.staff_allocation['security'],
            self.staff_allocation['checkin'],
            self.staff_allocation['baggage']
        ], dtype=np.float32)
        
        return obs
    
    def _get_passenger_flow(self, hour, dow):
        """Simulate passenger flow based on time"""
        hourly_pattern = {
            0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
            6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
            12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
            18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
        }
        dow_factor = 1.2 if dow in [4, 6] else 1.0  # Fri, Sun busier
        return hourly_pattern.get(hour, 200) * dow_factor * np.random.uniform(0.8, 1.2)
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply actions (adjust staff allocation)
        action_map = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
        
        for i, zone in enumerate(self.config['zones']):
            change = action_map[action[i]]
            new_staff = self.staff_allocation[zone] + change
            self.staff_allocation[zone] = np.clip(new_staff, self.min_staff, self.max_staff)
        
        # Simulate queue dynamics
        passenger_flow = self._get_passenger_flow(self.hour, self.day_of_week)
        
        for zone in self.config['zones']:
            # Service rate based on staff
            service_rate = self.staff_allocation[zone] * 15  # 15 passengers per staff per interval
            
            # Arrival rate proportional to passenger flow
            arrival_rates = {'security': 0.8, 'checkin': 0.6, 'baggage': 0.4}
            arrivals = passenger_flow * arrival_rates[zone] / 4  # Per 15 minutes
            
            # Update queue
            self.queue_lengths[zone] = max(0, self.queue_lengths[zone] + arrivals - service_rate)
            
            # Update wait time (Little's Law approximation)
            if service_rate > 0:
                self.wait_times[zone] = self.queue_lengths[zone] / service_rate * 15  # minutes
            else:
                self.wait_times[zone] = 60  # Cap at 60 min
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Update time
        self.current_step += 1
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day_of_week = (self.day_of_week + 1) % 7
        
        # Check if episode is done
        terminated = self.current_step >= self.config['max_steps']
        truncated = False
        
        # Store stats
        self.episode_stats['wait_times'].append(dict(self.wait_times))
        self.episode_stats['rewards'].append(reward)
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        """Calculate reward based on wait times and staff costs"""
        reward = 0
        
        # Penalty for exceeding target wait times
        for zone in self.config['zones']:
            target = self.config['target_wait_times'][zone]
            actual = self.wait_times[zone]
            
            if actual > target:
                # Quadratic penalty for exceeding target
                reward -= self.config['wait_time_penalty'] * ((actual - target) / target) ** 2
            else:
                # Small bonus for being under target
                reward += 2
        
        # Staff cost penalty
        total_staff = sum(self.staff_allocation.values())
        staff_cost = total_staff * self.config['staff_cost_per_hour'] / 4  # Per 15-min interval
        reward -= staff_cost / 100  # Normalize cost impact
        
        # Bonus for balanced allocation
        staff_values = list(self.staff_allocation.values())
        if max(staff_values) - min(staff_values) < 5:
            reward += 1
        
        return reward
    
    def render(self, mode='human'):
        """Render current state"""
        print(f"\n--- Step {self.current_step} | Hour: {self.hour:02d}:00 ---")
        print(f"Staff Allocation: {self.staff_allocation}")
        print(f"Queue Lengths: {self.queue_lengths}")
        print(f"Wait Times: {self.wait_times}")
        print(f"Cumulative Reward: {self.total_reward:.2f}")


class StaffSchedulerAgent:
    """
    RL Agent for staff scheduling using PPO or A2C
    """
    
    def __init__(self, model_type='PPO', model_path=None):
        self.env = DummyVecEnv([lambda: AirportStaffSchedulingEnv()])
        self.model_type = model_type
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            if model_type == 'PPO':
                self.model = PPO('MlpPolicy', self.env, verbose=1,
                               learning_rate=3e-4, n_steps=2048, batch_size=64)
            else:
                self.model = A2C('MlpPolicy', self.env, verbose=1,
                               learning_rate=7e-4, n_steps=5)
    
    def train(self, total_timesteps=100000):
        """Train the agent"""
        print(f"Training {self.model_type} agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        return self
    
    def predict(self, observation):
        """Get optimal action for given observation"""
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def get_schedule_recommendation(self, current_state):
        """
        Get staff scheduling recommendation based on current state
        
        Args:
            current_state: dict with keys: hour, dow, passenger_flow, 
                          queue_lengths, wait_times, current_staff
        
        Returns:
            dict with recommended staff allocation
        """
        obs = np.array([
            current_state.get('hour', 12),
            current_state.get('day_of_week', 0),
            current_state.get('passenger_flow', 400),
            current_state.get('security_queue', 30),
            current_state.get('checkin_queue', 20),
            current_state.get('baggage_queue', 15),
            current_state.get('security_wait', 15),
            current_state.get('checkin_wait', 8),
            current_state.get('baggage_wait', 10),
            current_state.get('security_staff', 8),
            current_state.get('checkin_staff', 10),
            current_state.get('baggage_staff', 5)
        ], dtype=np.float32)
        
        action = self.predict(obs.reshape(1, -1))
        action_map = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
        
        zones = ['security', 'checkin', 'baggage']
        current_staff = {
            'security': current_state.get('security_staff', 8),
            'checkin': current_state.get('checkin_staff', 10),
            'baggage': current_state.get('baggage_staff', 5)
        }
        
        recommendations = {}
        for i, zone in enumerate(zones):
            change = action_map[action[0][i]]
            new_staff = np.clip(current_staff[zone] + change, 2, 20)
            recommendations[zone] = {
                'current': current_staff[zone],
                'recommended': int(new_staff),
                'change': int(new_staff - current_staff[zone])
            }
        
        return recommendations
    
    def save(self, path):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model"""
        if self.model_type == 'PPO':
            self.model = PPO.load(path, env=self.env)
        else:
            self.model = A2C.load(path, env=self.env)
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    # Training example
    print("Creating Staff Scheduler RL Agent...")
    agent = StaffSchedulerAgent(model_type='PPO')
    
    print("\nTraining agent...")
    agent.train(total_timesteps=50000)
    
    # Save model
    agent.save('../saved_models/staff_scheduler/model')
    
    # Test recommendation
    print("\n--- Testing Recommendation ---")
    test_state = {
        'hour': 8,
        'day_of_week': 4,  # Friday
        'passenger_flow': 600,
        'security_queue': 50,
        'checkin_queue': 30,
        'baggage_queue': 20,
        'security_wait': 20,
        'checkin_wait': 10,
        'baggage_wait': 12,
        'security_staff': 8,
        'checkin_staff': 10,
        'baggage_staff': 5
    }
    
    recommendations = agent.get_schedule_recommendation(test_state)
    print("\nStaff Recommendations:")
    for zone, rec in recommendations.items():
        print(f"  {zone}: {rec['current']} -> {rec['recommended']} (change: {rec['change']:+d})")
