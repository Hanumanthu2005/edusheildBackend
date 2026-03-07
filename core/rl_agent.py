"""
EduShield RL-CNN Agent
=======================
Adaptive cheating detection using Deep Q-Learning with a 1D CNN.
Integrates directly with your existing Flask app (app.py).

State  : [yolo_conf, pose_ratio, audio_thresh_norm, violation_encoded]
Actions: 0=YOLO+  1=YOLO-  2=Pose+  3=Pose-  4=Audio+  5=Audio-  6=DoNothing
Reward : +1 (admin confirms cheat)  |  -1 (admin marks false positive)
"""

import os
import random
import numpy as np
from collections import deque

# TensorFlow is optional — falls back to numpy-only linear model
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[RL-Agent] TensorFlow not found — using lightweight numpy fallback model.")

import pickle

# ─── Constants ────────────────────────────────────────────────────────────────

ACTIONS = {
    0: "YOLO_THRESH_UP",
    1: "YOLO_THRESH_DOWN",
    2: "POSE_THRESH_UP",
    3: "POSE_THRESH_DOWN",
    4: "AUDIO_THRESH_UP",
    5: "AUDIO_THRESH_DOWN",
    6: "DO_NOTHING",
}

VIOLATION_MAP = {
    "phone_detected":          0.10,
    "book_detected":           0.20,
    "multiple_faces":          0.30,
    "looking_away":            0.45,
    "no_face":                 0.55,
    "tab_switch":              0.65,
    "fullscreen_exit":         0.75,
    "loud_noise":              0.85,
    "suspicious_audio_detected": 1.0,
}

STATE_SIZE  = 4
ACTION_SIZE = 7
WEIGHT_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
WEIGHT_PATH = os.path.join(WEIGHT_DIR, "rl_agent.h5")
FALLBACK_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "rl_agent_numpy.pkl")


# ─── Lightweight NumPy fallback (no TF needed) ────────────────────────────────

class _LinearQModel:
    """
    Linear Q-function: Q(s, a) = W_a · s + b_a
    Works without TensorFlow. Suitable for small state spaces.
    """
    def __init__(self, state_size, action_size):
        self.W = np.random.randn(action_size, state_size) * 0.01
        self.b = np.zeros(action_size)

    def predict(self, state: np.ndarray) -> np.ndarray:
        s = state.flatten()
        return self.W @ s + self.b

    def fit_step(self, state, action, target, lr=0.001):
        s = state.flatten()
        q = self.predict(s)
        err = target - q[action]
        self.W[action] += lr * err * s
        self.b[action] += lr * err

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"W": self.W, "b": self.b}, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.W = d["W"]
        self.b = d["b"]


# ─── Main RL-CNN Agent ────────────────────────────────────────────────────────

class RLCNNAgent:
    def __init__(self):
        os.makedirs(WEIGHT_DIR, exist_ok=True)

        self.state_size  = STATE_SIZE
        self.action_size = ACTION_SIZE

        # Replay memory
        self.memory   = deque(maxlen=2000)

        # Hyperparameters
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size    = 32

        # Dynamic detection thresholds — these are what the RL agent tunes
        self.thresholds = {
            "yolo":  0.50,   # YOLO object-detection confidence cutoff
            "pose":  0.25,   # Pose/gaze ratio cutoff
            "audio": 800.0,  # Audio amplitude threshold
        }

        # Feedback stats for accuracy tracking
        self.stats = {
            "total_feedback":  0,
            "true_positives":  0,
            "false_positives": 0,
            "accuracy_history": [],
        }

        # Build or load model
        self._use_tf = TF_AVAILABLE
        if self._use_tf:
            self.model = self._build_tf_model()
            if os.path.exists(WEIGHT_PATH):
                try:
                    self.model = load_model(WEIGHT_PATH)
                    self.epsilon = self.epsilon_min
                    print("[RL-CNN] Loaded TF weights from", WEIGHT_PATH)
                except Exception as e:
                    print(f"[RL-CNN] Could not load TF weights: {e}")
        else:
            self.model = _LinearQModel(STATE_SIZE, ACTION_SIZE)
            if os.path.exists(FALLBACK_WEIGHT_PATH):
                try:
                    self.model.load(FALLBACK_WEIGHT_PATH)
                    self.epsilon = self.epsilon_min
                    print("[RL-CNN] Loaded numpy weights from", FALLBACK_WEIGHT_PATH)
                except Exception as e:
                    print(f"[RL-CNN] Could not load numpy weights: {e}")

        # Load persisted stats if they exist
        self._load_stats()

    # ── Model Architecture ────────────────────────────────────────────────────

    def _build_tf_model(self):
        model = Sequential([
            Reshape((self.state_size, 1), input_shape=(self.state_size,)),
            Conv1D(filters=16, kernel_size=2, activation="relu"),
            Flatten(),
            Dense(24, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # ── State Encoding ────────────────────────────────────────────────────────

    def _encode_state(self, violation_type: str) -> np.ndarray:
        """Normalize thresholds + violation type into a 4D state vector."""
        v_encoded = VIOLATION_MAP.get(violation_type, 0.0)
        state = [
            self.thresholds["yolo"],
            self.thresholds["pose"],
            self.thresholds["audio"] / 2000.0,
            v_encoded,
        ]
        return np.reshape(state, [1, self.state_size])

    # ── Action Selection (ε-greedy) ───────────────────────────────────────────

    def _act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if self._use_tf:
            q_vals = self.model.predict(state, verbose=0)
            return int(np.argmax(q_vals[0]))
        else:
            q_vals = self.model.predict(state)
            return int(np.argmax(q_vals))

    # ── Threshold Mutation ────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        step       = 0.05
        audio_step = 100.0
        if   action == 0: self.thresholds["yolo"]  = min(0.85, self.thresholds["yolo"]  + step)
        elif action == 1: self.thresholds["yolo"]  = max(0.30, self.thresholds["yolo"]  - step)
        elif action == 2: self.thresholds["pose"]  = min(0.50, self.thresholds["pose"]  + step)
        elif action == 3: self.thresholds["pose"]  = max(0.10, self.thresholds["pose"]  - step)
        elif action == 4: self.thresholds["audio"] = min(2000.0, self.thresholds["audio"] + audio_step)
        elif action == 5: self.thresholds["audio"] = max(300.0,  self.thresholds["audio"] - audio_step)
        # action == 6 → Do Nothing

    # ── Experience Replay ─────────────────────────────────────────────────────

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            if self._use_tf:
                next_q  = self.model.predict(next_state, verbose=0)[0]
                target  = reward + self.gamma * np.amax(next_q)
                target_f = self.model.predict(state, verbose=0)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            else:
                next_q = self.model.predict(next_state)
                target = reward + self.gamma * np.amax(next_q)
                self.model.fit_step(state, action, target, self.learning_rate)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save_model(self):
        if self._use_tf:
            self.model.save(WEIGHT_PATH)
        else:
            self.model.save(FALLBACK_WEIGHT_PATH)

    def _save_stats(self):
        path = os.path.join(WEIGHT_DIR, "rl_stats.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "stats":      self.stats,
                "thresholds": self.thresholds,
                "epsilon":    self.epsilon,
            }, f)

    def _load_stats(self):
        path = os.path.join(WEIGHT_DIR, "rl_stats.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    d = pickle.load(f)
                self.stats      = d.get("stats",      self.stats)
                self.thresholds = d.get("thresholds", self.thresholds)
                self.epsilon    = d.get("epsilon",    self.epsilon)
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def update_thresholds(self, violation_type: str, feedback: int) -> dict:
        """
        Called by Flask when admin submits feedback.

        Parameters
        ----------
        violation_type : str   e.g. "looking_away"
        feedback       : int   +1 = confirmed cheat  |  -1 = false positive

        Returns
        -------
        dict  Updated thresholds + learning stats
        """
        # 1. Encode current state
        state = self._encode_state(violation_type)

        # 2. Agent picks an action
        action = self._act(state)

        # 3. Apply the action → mutate thresholds
        self._apply_action(action)

        # 4. Encode new state after mutation
        next_state = self._encode_state(violation_type)

        # 5. Store transition (reward = admin feedback)
        self.memory.append((state, action, float(feedback), next_state))

        # 6. Train on replay buffer
        self._replay()

        # 7. Update accuracy stats
        self.stats["total_feedback"] += 1
        if feedback == 1:
            self.stats["true_positives"] += 1
        else:
            self.stats["false_positives"] += 1
        self._record_accuracy()

        # 8. Persist weights + stats
        self._save_model()
        self._save_stats()

        tp  = self.stats["true_positives"]
        fp  = self.stats["false_positives"]
        acc = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0

        print(
            f"[RL-CNN] Feedback={'+1' if feedback>0 else '-1'} | "
            f"Action={ACTIONS[action]} | "
            f"YOLO={self.thresholds['yolo']:.2f} | "
            f"Pose={self.thresholds['pose']:.2f} | "
            f"Audio={self.thresholds['audio']:.0f} | "
            f"Accuracy={acc:.1%}"
        )

        return {
            "thresholds":  dict(self.thresholds),
            "action_taken": ACTIONS[action],
            "accuracy":    acc,
            "epsilon":     round(self.epsilon, 4),
            "total_feedback": self.stats["total_feedback"],
        }

    def _record_accuracy(self):
        tp  = self.stats["true_positives"]
        fp  = self.stats["false_positives"]
        acc = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        self.stats["accuracy_history"].append(acc)
        # Keep last 50 snapshots
        self.stats["accuracy_history"] = self.stats["accuracy_history"][-50:]

    def get_stats(self) -> dict:
        tp  = self.stats["true_positives"]
        fp  = self.stats["false_positives"]
        acc = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        return {
            "thresholds":       dict(self.thresholds),
            "total_feedback":   self.stats["total_feedback"],
            "true_positives":   tp,
            "false_positives":  fp,
            "accuracy":         acc,
            "accuracy_history": self.stats["accuracy_history"],
            "epsilon":          round(self.epsilon, 4),
        }


# ─── Singleton instance imported by Flask ────────────────────────────────────
rl_optimizer = RLCNNAgent()