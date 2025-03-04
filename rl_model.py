import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque

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
ACTION_SIZE = 2  # 0: Standard, 1: Increase

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
    if 6 <= hour <= 10 or 17 <= hour <= 21:  # Peak hours
        return 2 if action == 1 else -1  
    else:
        return 1 if action == 0 else -0.5  

# Streamlit UI
st.title("🔋 EV Charging Pricing Adjustment")

# User input for current hour
hour = st.number_input("Enter the current hour (0-23)", min_value=0, max_value=23, step=1)

if st.button("Calculate Price Adjustment"):
    reward = get_reward(1, hour)  # Action = 1 (increase)
    percentage = 1.2 if reward > 0 else 1.0  # Adjust price by 1.2% or keep it standard

    st.success(f"🔹 Current Hour: **{hour}**")
    st.success(f"📊 Charging Price Adjustment: **{percentage}x**")
