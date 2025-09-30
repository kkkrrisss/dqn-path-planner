import numpy as np
import matplotlib.pyplot as plt
from config import (
    GRID_WIDTH, GRID_HEIGHT, START_POS, GOAL_POS, OBSTACLES,
    REWARD_GOAL, REWARD_COLLISION, REWARD_STEP, VISION_RADIUS
)

class DroneEnv:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        self.obstacles = set(OBSTACLES)
        self.vision_radius = VISION_RADIUS
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.done = False
        return self.get_local_observation(self.vision_radius)

    def get_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        if self.done:
            raise Exception("Episode has ended. Call reset().")
        x, y = self.agent_pos
        if action == 0:    # up
            new_pos = (x, y - 1)
        elif action == 1:  # right
            new_pos = (x + 1, y)
        elif action == 2:  # down
            new_pos = (x, y + 1)
        elif action == 3:  # left
            new_pos = (x - 1, y)
        else:
            raise ValueError("Invalid action")

        if (
            0 <= new_pos[0] < self.width and
            0 <= new_pos[1] < self.height and
            new_pos not in self.obstacles
        ):
            self.agent_pos = new_pos
            if self.agent_pos == self.goal_pos:
                self.done = True
                reward = REWARD_GOAL
            else:
                reward = REWARD_STEP
        else:
            reward = REWARD_COLLISION
            self.done = True

        return self.get_local_observation(self.vision_radius), reward, self.done

    def get_local_observation(self, vision_radius=1):
        x, y = self.agent_pos
        obs = []
        for dy in range(-vision_radius, vision_radius + 1):
            row = []
            for dx in range(-vision_radius, vision_radius + 1):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    row.append(1)
                elif (nx, ny) in self.obstacles:
                    row.append(1)
                elif (nx, ny) == self.start_pos:
                    row.append(2)
                elif (nx, ny) == self.goal_pos:
                    row.append(3)
                elif (nx, ny) == self.agent_pos:
                    row.append(4)
                else:
                    row.append(0)
            obs.append(tuple(row))
        return tuple(obs)

    def render(self, ax=None, show=True, title="", path=None):
        grid = np.ones((self.height, self.width, 3), dtype=np.float32)
        for (x, y) in self.obstacles:
            grid[y, x] = [0, 0, 0]
        sx, sy = self.start_pos
        grid[sy, sx] = [0.0, 0.0, 1.0]
        gx, gy = self.goal_pos
        grid[gy, gx] = [0.0, 1.0, 0.0]
        if path is not None:
            for (x, y) in path:
                if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                    grid[y, x] = [1.0, 0.6, 0.6]
        x, y = self.agent_pos
        grid[y, x] = [1.0, 0.0, 0.0]
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots()
        ax.clear()
        ax.imshow(grid, interpolation='none')
        ax.set_title(title)
        ax.axis('off')
        for i in range(self.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.7)
        for j in range(self.height + 1):
            ax.axhline(j - 0.5, color='gray', linewidth=0.7)
        plt.pause(0.001)
        if show:
            plt.show(block=False)
        return ax

    @property
    def agent_true_pos(self):
        return self.agent_pos