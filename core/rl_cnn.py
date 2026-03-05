import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

class RLCNNAgent:
    def __init__(self):
        # Define the path where the RL-CNN will save its learned weights
        self.weight_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'rl_agent.h5')
        
        # State: [yolo_conf, pose_ratio, audio_thresh_normalized, violation_encoded]
        self.state_size = 4 
        
        # Actions: 0: YOLO+, 1: YOLO-, 2: Pose+, 3: Pose-, 4: Audio+, 5: Audio-, 6: Do Nothing
        self.action_size = 7 
        
        # RL Hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate for future rewards
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # System Thresholds (Starting points)
        self.thresholds = {
            "yolo": 0.50,
            "pose": 0.25,
            "audio": 800.0
        }
        
        # Violation Encoding Map for the Neural Network
        self.violation_map = {
            "unauthorized_device": 0.0,
            "multiple_faces": 0.25,
            "looking_left": 0.50,
            "looking_right": 0.50,
            "suspicious_audio_detected": 1.0
        }

        # Build or Load the RL-CNN Model
        self.model = self._build_model()
        if os.path.exists(self.weight_path):
            print("[RL-CNN] Loading existing learned weights...")
            try:
                self.model = load_model(self.weight_path)
                self.epsilon = self.epsilon_min # Less exploration if already trained
            except Exception as e:
                print(f"[RL-CNN ERROR] Could not load weights: {e}")

    def _build_model(self):
        """Builds the 1D CNN architecture specified in the research paper."""
        model = Sequential()
        # Input shape requires reshaping for Conv1D: (batch_size, steps, features)
        model.add(Reshape((self.state_size, 1), input_shape=(self.state_size,)))
        
        # CNN Feature Extraction
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        model.add(Flatten())
        
        # Dense Decision Layers
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        
        # Output Layer: Linear activation for Q-values
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _get_current_state(self, violation_type):
        """Normalizes the current thresholds and the active violation into a 1D tensor."""
        v_encoded = self.violation_map.get(violation_type, 0.0)
        state = [
            self.thresholds["yolo"],
            self.thresholds["pose"],
            self.thresholds["audio"] / 2000.0, # Normalize audio to keep it bounded 0-1
            v_encoded
        ]
        return np.reshape(state, [1, self.state_size])

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0]) # Exploit learned Q-values

    def _apply_action(self, action):
        """Translates the network's predicted action into actual threshold adjustments."""
        step = 0.05
        audio_step = 100.0
        
        if action == 0: self.thresholds["yolo"] = min(0.85, self.thresholds["yolo"] + step)
        elif action == 1: self.thresholds["yolo"] = max(0.30, self.thresholds["yolo"] - step)
        elif action == 2: self.thresholds["pose"] = min(0.50, self.thresholds["pose"] + step)
        elif action == 3: self.thresholds["pose"] = max(0.10, self.thresholds["pose"] - step)
        elif action == 4: self.thresholds["audio"] = min(2000.0, self.thresholds["audio"] + audio_step)
        elif action == 5: self.thresholds["audio"] = max(300.0, self.thresholds["audio"] - audio_step)
        # Action 6 is "Do Nothing"

    def remember(self, state, action, reward, next_state):
        """Stores the experience in memory for batch training."""
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size=32):
        """Trains the CNN on past experiences to optimize Q-values."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            # Deep Q-Learning Bellman Equation
            target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Save the brain periodically
        self.model.save(self.weight_path)
        print(f"[RL-CNN] Network trained. Weights saved to {self.weight_path}")

    def update_thresholds(self, violation_type, feedback):
        """
        The main API method called by Flask when the admin clicks +1 or -1.
        """
        # 1. Observe current state
        state = self._get_current_state(violation_type)
        
        # 2. Decide on an action
        action = self.act(state)
        
        # 3. Execute the action
        self._apply_action(action)
        
        # 4. Observe new state
        next_state = self._get_current_state(violation_type)
        
        # 5. Store the experience (Feedback is the Reward: +1 or -1)
        self.remember(state, action, feedback, next_state)
        
        # 6. Train the network in the background
        self.replay()
        
        print(f"[RL-CNN] Updated Params -> YOLO: {self.thresholds['yolo']:.2f} | Audio: {self.thresholds['audio']}")
        return self.thresholds

# Global instance to be imported by the Flask app
rl_optimizer = RLCNNAgent()