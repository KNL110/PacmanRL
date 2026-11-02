"""Maze-chase Gymnasium environment.

Two observation modes:

* ``"features"`` (default, for tabular methods): a small MultiDiscrete vector
      [wall_up, wall_down, wall_left, wall_right,        0/1 each
       danger_up, danger_down, danger_left, danger_right, 0/1 each
       pellet_dir,                                       0-3, or 4 = none
       edible_dir]                                       0-3, or 4 = none
  "danger_X" means moving X gets closer to a non-frightened ghost that is
  within `dangerRadius`. "pellet_dir" / "edible_dir" are the first move on
  the shortest path to the nearest pellet / frightened ghost.
  Use ``tuple(obs)`` as a Q-table key (16 * 16 * 5 * 5 = 6400 states max).

* ``"grid"`` (for neural nets): float32 array of shape (8, H, W) with layers
  walls, pellets, power pellets, agent, one per ghost, and frightened ghosts
  (valued by the fraction of frightened time left).
"""

import time
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ghosts import GHOST_TYPES, Ghost
from .grid import Grid
from .layout import DEFAULT_LAYOUT, Pos, parseLayout


@dataclass(frozen=True)
class Rewards:
    step: float = -0.01
    pellet: float = 1.0
    powerPellet: float = 5.0
    eatGhost: float = 20.0
    death: float = -50.0
    clear: float = 100.0


@dataclass(frozen=True)
class Info:
    """Per-step info. Gymnasium requires a dict, so it is returned via `asdict`."""

    score: float
    steps: int
    pelletsLeft: int
    frightenedLeft: int
    outcome: str | None = None  # "death", "clear", or None while running


NONE_DIR = 4


class MazeChaseEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 8}

    def __init__(
        self,
        layout: str = DEFAULT_LAYOUT,
        obsMode: str = "features",
        render_mode: str | None = None,
        maxSteps: int = 500,
        rewards: Rewards | dict[str, float] = Rewards(),
        frightenedSteps: int = 20,
        respawnWait: int = 10,
        releaseDelays: tuple[int, ...] = (0, 5, 10),
        dangerRadius: int = 3,
    ):
        if obsMode not in ("features", "grid"):
            raise ValueError(f"unknown obsMode {obsMode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")

        self.layout = parseLayout(layout)
        self.grid = Grid(self.layout)
        self.obsMode = obsMode
        self.render_mode = render_mode
        self.maxSteps = maxSteps
        # A dict (e.g. from config.yaml) overrides only the fields it names.
        self.rewards = Rewards(**rewards) if isinstance(rewards, dict) else rewards
        self.frightenedSteps = frightenedSteps
        self.respawnWait = respawnWait
        self.dangerRadius = dangerRadius

        if len(self.layout.ghostStarts) > len(GHOST_TYPES):
            raise ValueError(f"layout has more than {len(GHOST_TYPES)} ghost spawns")
        self.ghosts: list[Ghost] = [
            cls(home, releaseDelays[i] if i < len(releaseDelays) else 0)
            for i, (cls, home) in enumerate(zip(GHOST_TYPES, self.layout.ghostStarts))
        ]

        self.gridShape = (4 + len(self.ghosts) + 1, *self.layout.shape)
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        if obsMode == "features":
            self.observation_space = spaces.MultiDiscrete([2] * 8 + [5, 5])
        else:
            self.observation_space = spaces.Box(0.0, 1.0, shape=self.gridShape, dtype=np.float32)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.pellets = self.layout.pellets.copy()
        self.powerPellets = self.layout.powerPellets.copy()
        self.agent: Pos = self.layout.agentStart
        for ghost in self.ghosts:
            ghost.reset()
        self.frightenedLeft = 0
        self.steps = 0
        self.score = 0.0
        if self.render_mode == "human":
            self.render()
        return self.obs(), asdict(self.info())

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        self.steps += 1
        reward = self.rewards.step

        # 1. Agent moves, then collides with ghosts still on their old tiles.
        #    This also catches head-on swaps (agent and ghost trading tiles).
        self.agent = self.grid.move(self.agent, int(action))
        dead, r = self.collide()
        reward += r

        # 2. Agent eats whatever is on its tile.
        if not dead:
            reward += self.eat()

        # 3. Ghosts move (frightened ones only on even steps), collide again.
        if not dead:
            power = self.powerPositions()
            for ghost in self.ghosts:
                if ghost.frightened and self.steps % 2:
                    continue
                ghost.step(self.grid, self.agent, power, self.np_random)
            dead, r = self.collide()
            reward += r

        # 4. Frightened timer.
        if self.frightenedLeft > 0:
            self.frightenedLeft -= 1
            if self.frightenedLeft == 0:
                for ghost in self.ghosts:
                    if ghost.frightened:
                        ghost.frightened = False
                        ghost.direction = None

        cleared = not self.pellets.any() and not self.powerPellets.any()
        if dead:
            reward += self.rewards.death
        elif cleared:
            reward += self.rewards.clear

        terminated = dead or cleared
        truncated = not terminated and self.steps >= self.maxSteps
        self.score += reward

        if self.render_mode == "human":
            self.render()
        info = self.info("death" if dead else "clear" if cleared else None)
        return self.obs(), reward, terminated, truncated, asdict(info)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode")
            return None
        text = self.renderText()
        if self.render_mode == "human":
            # Cursor to top-left + clear screen, so each frame replaces the last.
            print("\033[H\033[2J" + text, flush=True)
            time.sleep(1 / self.metadata["render_fps"])
            return None
        return text

    # ------------------------------------------------------------- internals

    def collide(self) -> tuple[bool, float]:
        reward = 0.0
        for ghost in self.ghosts:
            if not ghost.active or ghost.pos != self.agent:
                continue
            if ghost.frightened:
                ghost.sendHome(self.respawnWait)
                reward += self.rewards.eatGhost
            else:
                return True, reward
        return False, reward

    def eat(self) -> float:
        r, c = self.agent
        if self.pellets[r, c]:
            self.pellets[r, c] = False
            return self.rewards.pellet
        if self.powerPellets[r, c]:
            self.powerPellets[r, c] = False
            self.frightenedLeft = self.frightenedSteps
            for ghost in self.ghosts:
                if ghost.active:
                    ghost.frightened = True
                    ghost.direction = None  # mode change: allowed to turn around
            return self.rewards.powerPellet
        return 0.0

    def powerPositions(self) -> list[Pos]:
        return [(int(r), int(c)) for r, c in zip(*np.nonzero(self.powerPellets))]

    def firstMoveTowards(self, targets: list[Pos]) -> int:
        """First action on a shortest path from the agent to the nearest target."""
        if not targets:
            return NONE_DIR
        targetIdx = [self.grid.index[t] for t in targets]
        bestAction, bestDist = NONE_DIR, None
        for action, pos in self.grid.neighbors(self.agent):
            d = int(self.grid.dist[self.grid.index[pos], targetIdx].min())
            if bestDist is None or d < bestDist:
                bestAction, bestDist = action, d
        return bestAction

    def obs(self) -> np.ndarray:
        if self.obsMode == "grid":
            return self.gridObs()

        walls = [1] * 4
        danger = [0] * 4
        threats = [
            (g.pos, self.grid.distance(self.agent, g.pos))
            for g in self.ghosts
            if g.active and not g.frightened
        ]
        threats = [(p, d) for p, d in threats if d <= self.dangerRadius]
        for action, pos in self.grid.neighbors(self.agent):
            walls[action] = 0
            if any(self.grid.distance(pos, p) < d for p, d in threats):
                danger[action] = 1

        food = [(int(r), int(c)) for r, c in zip(*np.nonzero(self.pellets | self.powerPellets))]
        edible = [g.pos for g in self.ghosts if g.active and g.frightened]
        return np.array(
            walls + danger + [self.firstMoveTowards(food), self.firstMoveTowards(edible)],
            dtype=np.int64,
        )

    def gridObs(self) -> np.ndarray:
        obs = np.zeros(self.gridShape, dtype=np.float32)
        obs[0] = self.layout.walls
        obs[1] = self.pellets
        obs[2] = self.powerPellets
        obs[3][self.agent] = 1.0
        for i, ghost in enumerate(self.ghosts):
            if ghost.active:
                obs[4 + i][ghost.pos] = 1.0
                if ghost.frightened:
                    obs[-1][ghost.pos] = self.frightenedLeft / self.frightenedSteps
        return obs

    def info(self, outcome: str | None = None) -> Info:
        return Info(
            score=self.score,
            steps=self.steps,
            pelletsLeft=int(self.pellets.sum() + self.powerPellets.sum()),
            frightenedLeft=self.frightenedLeft,
            outcome=outcome,
        )

    def renderText(self) -> str:
        chars = np.where(self.layout.walls, "#", " ").astype("<U1")
        chars[self.pellets] = "."
        chars[self.powerPellets] = "o"
        for ghost in self.ghosts:
            if ghost.active:
                chars[ghost.pos] = ghost.symbol.lower() if ghost.frightened else ghost.symbol
        chars[self.agent] = "P"
        grid = "\n".join("".join(row) for row in chars)
        status = f"step {self.steps}  score {self.score:.2f}"
        if self.frightenedLeft:
            status += f"  frightened {self.frightenedLeft}"
        return f"{grid}\n{status}"
