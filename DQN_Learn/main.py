from environment import DroneEnv
from agent import DQNAgent
from config import (
    NUM_EPISODES, MAX_STEPS_PER_EPISODE, VISION_RADIUS,
    LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    REPLAY_BUFFER_SIZE, BATCH_SIZE, TARGET_UPDATE_FREQ
)
import numpy as np
import matplotlib.pyplot as plt

def preprocess_observation(obs):
    return np.array(obs, dtype=np.float32)

def train():
    env = DroneEnv()
    obs_shape = np.array(env.get_local_observation(VISION_RADIUS)).shape
    n_actions = len(env.get_actions())
    agent = DQNAgent(
        input_shape=obs_shape, n_actions=n_actions,
        learning_rate=LEARNING_RATE, gamma=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY,
        buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        device='cpu'
    )
    rewards_per_episode = []
    best_path = []
    best_steps = float('inf')

    plt.ion()
    fig, ax = plt.subplots()

    for episode in range(NUM_EPISODES):
        observation = env.reset()
        state = preprocess_observation(observation)
        total_reward = 0
        trajectory = [env.agent_true_pos]

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_observation, reward, done = env.step(action)
            next_state = preprocess_observation(next_observation)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            trajectory.append(env.agent_true_pos)
            total_reward += reward

            if done:
                break

        agent.update_epsilon()
        rewards_per_episode.append(total_reward)

        env.render(ax=ax, show=False, title=f"Эпизод {episode+1}")

        if env.agent_true_pos == env.goal_pos and len(trajectory) < best_steps:
            best_steps = len(trajectory)
            best_path = trajectory.copy()

    plt.ioff()
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel('Эпизод')
    plt.ylabel('Суммарное вознаграждение')
    plt.title('Динамика обучения')
    plt.show()

    agent.save()

    if best_path:
        fig2, ax2 = plt.subplots()
        env.agent_pos = env.start_pos
        env.render(ax=ax2, show=True, title="Лучший найденный путь", path=best_path)
        plt.show()
        print("Лучший найденный путь (координаты):")
        for coord in best_path:
            print(coord)
        print(f"Финальные координаты: {best_path[-1]}")
        print(f"Длина лучшего пути: {len(best_path)}")
    else:
        print("Дрон не нашёл путь до цели.")

if __name__ == "__main__":
    train()