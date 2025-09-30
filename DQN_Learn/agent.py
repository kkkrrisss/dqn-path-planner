import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self, input_shape, n_actions,
        learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay,
        buffer_size, batch_size, target_update_freq, device='cpu'
    ):
        self.n_actions = n_actions
        self.device = device
        self.model = DQN(input_shape, n_actions).to(device)
        self.target_model = DQN(input_shape, n_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learn_step = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_v = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.model(state_v)
        return int(torch.argmax(q_vals).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_v = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.device)

        state_action_values = self.model(states_v).gather(1, actions_v).squeeze()
        with torch.no_grad():
            next_state_values = self.target_model(next_states_v).max(1)[0]
            next_state_values[dones_v] = 0.0
            expected_q = rewards_v + self.gamma * next_state_values

        loss = nn.MSELoss()(state_action_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filename='dqn_agent.pth'):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename='dqn_agent.pth'):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())