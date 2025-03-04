from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
from datetime import datetime

app = Flask(__name__)

# Load dataset
df = pd.read_csv("charging_data.csv")

# Encode categorical features
df["Weather"] = df["Weather"].astype("category").cat.codes
df["Weekend"] = df["Weekend"].astype("category").cat.codes

# Convert DataFrame to NumPy array
data = df.to_numpy()

# Hyperparameters
EPISODES = 2000
GAMMA = 0.9
EPSILON = 0.9
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MEMORY = deque(maxlen=10000)  

# Define state and action sizes
STATE_SIZE = data.shape[1]
ACTION_SIZE = 2  

# Define Neural Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Reward function
def get_reward(action, hour):
    if 6 <= hour <= 10 or 17 <= hour <= 21:
        return 2 if action == 1 else -1  
    else:
        return 1 if action == 0 else -0.5  

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "EV Charging RL Model API is running"}), 200

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.json
        hour = data.get("hour")

        if hour is None or not isinstance(hour, int) or hour < 0 or hour > 23:
            return jsonify({"error": "Invalid hour. Must be an integer between 0 and 23."}), 400

        # Select state
        state = torch.tensor([hour, 80, 2, 0, 56], dtype=torch.float32).to(device)

        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            action = random.randint(0, ACTION_SIZE - 1)
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()

        # Compute reward
        reward = get_reward(action, hour)
        percentage = 1.2 if reward > 0 else 1.0  

        return jsonify({
            "hour": hour,
            "action": action,
            "reward": reward,
            "percentage": percentage,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
