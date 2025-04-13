import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class IMUDataGenerator:
    def __init__(self, sampling_rate=100, random_seed=42):
        """
        Initialize the IMU data generator
        
        Args:
            sampling_rate: Number of IMU readings per second (Hz)
            random_seed: Random seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        np.random.seed(random_seed)
        
        # Set default noise levels
        self.accel_noise_level = 0.05  # m/s^2
        self.gyro_noise_level = 0.02   # rad/s
        
    def generate_normal_driving(self, duration_seconds, speed_kmh=60):
        """
        Generate synthetic IMU data for normal driving
        
        Args:
            duration_seconds: Length of data to generate in seconds
            speed_kmh: Average vehicle speed in km/h
            
        Returns:
            DataFrame with synthetic IMU data
        """
        num_samples = int(duration_seconds * self.sampling_rate)
        t = np.linspace(0, duration_seconds, num_samples)
        
        # Convert speed to m/s
        speed_ms = speed_kmh / 3.6
        
        # Initialize arrays
        accel_x = np.zeros(num_samples)
        accel_y = np.zeros(num_samples)
        accel_z = np.ones(num_samples) * 9.81  # Gravity
        gyro_x = np.zeros(num_samples)
        gyro_y = np.zeros(num_samples)
        gyro_z = np.zeros(num_samples)
        
        # Add road irregularities - low frequency components
        road_freq = 0.2 + 0.3 * np.random.random()  # 0.2-0.5 Hz
        road_amp = 0.2 + 0.3 * np.random.random()   # 0.2-0.5 m/s^2
        accel_z += road_amp * np.sin(2 * np.pi * road_freq * t)
        
        # Add engine vibration - higher frequency
        engine_freq = 20 + 10 * np.random.random()  # 20-30 Hz
        engine_amp = 0.1 + 0.1 * np.random.random() # 0.1-0.2 m/s^2
        accel_x += engine_amp * np.sin(2 * np.pi * engine_freq * t)
        
        # Add random driving events
        self._add_random_driving_events(t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        
        # Add noise
        accel_x += np.random.normal(0, self.accel_noise_level, num_samples)
        accel_y += np.random.normal(0, self.accel_noise_level, num_samples)
        accel_z += np.random.normal(0, self.accel_noise_level, num_samples)
        gyro_x += np.random.normal(0, self.gyro_noise_level, num_samples)
        gyro_y += np.random.normal(0, self.gyro_noise_level, num_samples)
        gyro_z += np.random.normal(0, self.gyro_noise_level, num_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': t,
            'acc_x': accel_x,
            'acc_y': accel_y,
            'acc_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'label': 'normal'
        })
        
        return df
    
    def _add_random_driving_events(self, t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        """Add random normal driving events like turns, acceleration, braking"""
        num_samples = len(t)
        
        # Add several random events
        num_events = int(len(t) / self.sampling_rate / 10)  # Roughly one event every 10 seconds
        
        for _ in range(num_events):
            # Choose random start point
            start_idx = np.random.randint(0, num_samples - self.sampling_rate * 5)  # At least 5s from end
            
            # Choose random event type
            event_type = np.random.choice(['acceleration', 'braking', 'turn_left', 'turn_right'])
            
            # Event duration between 1-4 seconds
            duration_idx = np.random.randint(self.sampling_rate, self.sampling_rate * 4)
            end_idx = min(start_idx + duration_idx, num_samples)
            
            # Create event window with smooth start/end
            window = signal.windows.tukey(end_idx - start_idx, alpha=0.25)
            
            if event_type == 'acceleration':
                # Forward acceleration: positive accel_x
                magnitude = 1.0 + 1.0 * np.random.random()  # 1-2 m/s^2
                accel_x[start_idx:end_idx] += magnitude * window
                
            elif event_type == 'braking':
                # Braking: negative accel_x
                magnitude = 1.5 + 1.5 * np.random.random()  # 1.5-3 m/s^2
                accel_x[start_idx:end_idx] -= magnitude * window
                
            elif event_type == 'turn_left':
                # Left turn: positive accel_y (lateral) and gyro_z (yaw rate)
                accel_magnitude = 1.0 + 1.0 * np.random.random()  # 1-2 m/s^2
                gyro_magnitude = 0.1 + 0.2 * np.random.random()   # 0.1-0.3 rad/s
                
                accel_y[start_idx:end_idx] += accel_magnitude * window
                gyro_z[start_idx:end_idx] += gyro_magnitude * window
                
            elif event_type == 'turn_right':
                # Right turn: negative accel_y and gyro_z
                accel_magnitude = 1.0 + 1.0 * np.random.random()  # 1-2 m/s^2
                gyro_magnitude = 0.1 + 0.2 * np.random.random()   # 0.1-0.3 rad/s
                
                accel_y[start_idx:end_idx] -= accel_magnitude * window
                gyro_z[start_idx:end_idx] -= gyro_magnitude * window
