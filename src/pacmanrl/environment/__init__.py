import gymnasium as gym

from .env import Info, MazeChaseEnv, Rewards

gym.register(id="MazeChase-v0", entry_point="pacmanrl.environment.env:MazeChaseEnv")

__all__ = ["Info", "MazeChaseEnv", "Rewards"]
