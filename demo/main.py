import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
np.random.seed(120)  # Set a fixed random seed for reproducibility
start_pos = np.array([0, 0])  # UAV's starting position
goal_pos = np.array([100, 0])  # Goal position for the UAV
obstacles = np.array([[30, 20], [60, -10], [80, 15]])  # 3 obstacles in the environment
obstacle_radius = 5  # The radius of the obstacles (area around each obstacle)
delta_angle = 30  # ±30° heading control for UAV

# UAV State and Rewards
uav_pos = np.copy(start_pos)  # Copy the initial position of the UAV
uav_angle = 0  # UAV's initial heading (0 degrees, aligned with x-axis)
trajectory = [np.copy(uav_pos)]  # Store UAV's path as a list of positions
reward = 0  # Initialize reward


# Define the Distance function and Collision check
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)  # Euclidean distance between two points (p1, p2)


def check_collision(pos):
    """
    Checks if the UAV collides with any obstacles.
    Returns True if collision occurs, otherwise False.
    """
    for obs in obstacles:
        if distance(pos, obs) < obstacle_radius:  # If the UAV is within obstacle's radius
            return True
    return False


def step(action):
    """
    Takes an action, updates the UAV's state, and calculates the reward.
    Returns True if the episode is done (UAV reached goal or collided).
    """
    global uav_pos, uav_angle, reward

    # Update heading angle
    uav_angle += action  # Modify the UAV's heading based on action
    uav_angle = np.clip(uav_angle, -30, 30)  # Ensure angle is within [-30, 30] range

    # Move UAV: calculate the new position based on the step size and angle
    step_size = 7  # Step size for the UAV's movement (larger for faster movement)
    dx = step_size * np.cos(np.radians(uav_angle))  # Change in x-position
    dy = step_size * np.sin(np.radians(uav_angle))  # Change in y-position
    uav_pos[0] += dx  # Update UAV's x-position
    uav_pos[1] += dy  # Update UAV's y-position
    trajectory.append(np.copy(uav_pos))  # Append new position to the trajectory

    # Reward Calculation
    if distance(uav_pos, goal_pos) < 10:  # If the UAV is close enough to the goal (within 10 units)
        reward += 50  # Give a large reward for reaching the goal
        return True  # Episode is done
    elif check_collision(uav_pos):  # If UAV collides with an obstacle
        reward -= 20  # Give a penalty for collision
        return True  # Episode is done
    else:
        reward -= (0.01 * distance(uav_pos,
                                   goal_pos) + 0.1)  # Penalize based on distance to the goal Distance-based penalty to encourage the agent to move towards the goal
        return False  # Episode continues


# Define the Q-Network (Neural Network)
class QNetwork(nn.Module):
    """
    Neural network for approximating the Q-function in DQN.
    The network takes in a state and outputs Q-values for each action.
    """

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # First fully connected layer (input size -> 64 neurons)
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer (64 neurons -> 64 neurons)
        self.fc3 = nn.Linear(64, action_size)  # Output layer (64 neurons -> number of actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)  # Output Q-values for each action
        return x


# DQN Agent
class DQNAgent:
    """
    DQN Agent with experience replay and epsilon-greedy action selection.
    """

    def __init__(self, state_size, action_size):
        # Initialize state, action size, epsilon for exploration, discount factor, etc.
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate (start with high exploration)
        self.epsilon_decay = 0.995  # Decay factor for epsilon (slow exploration decrease)
        self.epsilon_min = 0.1  # Minimum value of epsilon
        self.gamma = 0.99  # Discount factor for future rewards
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.memory = deque(maxlen=2000)  # Memory buffer for experience replay (max 2000 experiences)
        self.batch_size = 64  # Batch size for training the model
        self.model = QNetwork(state_size, action_size)  # Q-network (model)
        self.target_model = QNetwork(state_size, action_size)  # Target model (copy of Q-network)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer
        self.update_target_model()  # Synchronize target model with Q-network

    def update_target_model(self):
        """Synchronizes the target model with the current Q-network model."""
        self.target_model.load_state_dict(self.model.state_dict())  # Copy Q-network weights to target model

    def act(self, state):
        """
        Chooses an action using epsilon-greedy policy.
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the best action based on Q-values (exploitation).
        """
        if np.random.rand() <= self.epsilon:  # Exploration
            return np.random.choice(self.action_size)  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        q_values = self.model(state)  # Get Q-values from the model
        return torch.argmax(q_values).item()  # Choose action with the highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in memory for replay."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Samples a batch of experiences from memory and performs Q-network training.
        The model learns from past experiences and adjusts its Q-values.
        """
        if len(self.memory) < self.batch_size:  # Not enough experiences to sample a batch
            return
        batch = random.sample(self.memory, self.batch_size)  # Sample a random batch
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Convert next state to tensor
            target = reward  # Start with the immediate reward
            if not done:  # If the episode is not finished, add future discounted reward
                target += self.gamma * torch.max(
                    self.target_model(next_state)).item()  # Use target model for future rewards
            target_f = self.model(state)  # Get Q-values for the current state from the Q-network
            target_f[0][action] = target  # Update the Q-value for the chosen action
            loss = nn.MSELoss()(target_f, self.model(state))  # Calculate the loss (mean squared error)
            self.optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Perform backpropagation
            self.optimizer.step()  # Update the model's weights

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# DQN Training Loop
agent = DQNAgent(state_size=2, action_size=3)  # Initialize agent with 2 state variables (x, y) and 3 possible actions
max_episodes = 2000  # Number of episodes for training
max_steps = 100  # Max steps per episode (number of actions per episode)
reward_history = []  # List to store total reward for each episode

# Mapping the action index to angles
action_map = {-30: 0, 0: 1, 30: 2, }  # Action mapping (index -> angle)

# Training loop
for episode in range(max_episodes):
    uav_pos = np.copy(start_pos)  # Reset UAV position at the start of each episode
    trajectory = [np.copy(uav_pos)]  # Start trajectory list
    uav_angle = 0  # Reset UAV angle to 0 at the start of each episode
    total_reward = 0  # Total reward for the episode
    for step_num in range(max_steps):
        state = np.array([uav_pos[0], uav_pos[1]])  # Get current state (UAV position)
        action_idx = agent.act(state)  # Choose action using agent's policy
        action = list(action_map.keys())[action_idx]  # Convert action index to corresponding angle
        done = step(action)  # Perform the chosen action and check if done
        next_state = np.array([uav_pos[0], uav_pos[1]])  # Get next state after action
        agent.remember(state, action, reward, next_state, done)  # Store the experience in memory
        total_reward += reward  # Add the reward to the total reward for this episode
        if done:  # If episode is finished (goal reached or collision occurred)
            break
    agent.update_target_model()  # Update the target model after each episode
    reward_history.append(total_reward)  # Store the total reward for the episode

# Visualization (Matplotlib for plotting and animation)
fig, ax = plt.subplots()  # Create a new figure for the plot
ax.set_xlim(-10, 110)  # Set x-axis limits
ax.set_ylim(-40, 40)  # Set y-axis limits
ax.set_xlabel("X Position")  # Label for x-axis
ax.set_ylabel("Y Position")  # Label for y-axis
ax.set_title("UAV Demo Using DQN")  # Title for the plot

# Draw Goal and Obstacles
ax.scatter(goal_pos[0], goal_pos[1], color='green', s=100, label="Goal")  # Plot goal position
for obs in obstacles:
    circle = plt.Circle(obs, obstacle_radius, color='red', alpha=0.5)  # Draw obstacles as circles
    ax.add_patch(circle)  # Add obstacles to the plot
ax.legend()  # Show legend

# UAV Path Animation
uav_marker, = ax.plot([], [], 'bo-', markersize=8, label="UAV")  # Create marker for UAV path


def update(frame):
    """
    Updates the UAV's path for animation.
    """
    if frame < len(trajectory):
        uav_marker.set_data([p[0] for p in trajectory[:frame]],
                            [p[1] for p in trajectory[:frame]])  # Update UAV position
    return uav_marker,


# Animation
ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=200, blit=True)
plt.show()  # Show the animation

print(f"Final Reward: {total_reward}")  # Print the final reward for the episode
