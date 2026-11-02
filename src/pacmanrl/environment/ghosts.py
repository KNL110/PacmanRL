"""Ghost policies.

All ghosts move one tile per step and never reverse unless at a dead end or
their mode just changed. Each subclass only decides *where* it wants to go;
frightened ghosts ignore that and flee from the agent instead.
"""

import numpy as np
from .grid import OPPOSITE, Grid
from .layout import Pos


class Ghost:
    name = "ghost"
    symbol = "G"

    def __init__(self, home: Pos, releaseDelay: int = 0):
        self.home = home
        self.releaseDelay = releaseDelay
        self.reset()

    def reset(self) -> None:
        self.pos = self.home
        self.direction: int | None = None  # None = free to reverse next move
        self.wait = self.releaseDelay      # steps left before it can move
        self.frightened = False

    @property
    def active(self) -> bool:
        return self.wait == 0

    def sendHome(self, wait: int) -> None:
        self.pos = self.home
        self.direction = None
        self.wait = wait
        self.frightened = False

    def target(self, grid: Grid, agent: Pos, powerPellets: list[Pos], rng: np.random.Generator) -> Pos | None:
        """Tile to head towards, or None to wander randomly."""
        raise NotImplementedError

    def step(self, grid: Grid, agent: Pos, powerPellets: list[Pos], rng: np.random.Generator) -> None:
        if not self.active:
            self.wait -= 1
            return

        options = grid.neighbors(self.pos)
        if self.direction is not None and len(options) > 1:
            options = [o for o in options if o[0] != OPPOSITE[self.direction]]

        if self.frightened:
            scores = [-grid.distance(p, agent) for _, p in options]
        else:
            goal = self.target(grid, agent, powerPellets, rng)
            if goal is None:
                scores = [0] * len(options)
            else:
                scores = [grid.distance(p, goal) for _, p in options]

        best = min(scores)
        choices = [o for o, s in zip(options, scores) if s == best]
        self.direction, self.pos = choices[rng.integers(len(choices))]


class Hound(Ghost):
    """Chases the agent directly, with an occasional random move."""

    name, symbol = "hound", "H"

    def __init__(self, home: Pos, releaseDelay: int = 0, epsilon: float = 0.3):
        super().__init__(home, releaseDelay)
        self.epsilon = epsilon

    def target(self, grid, agent, powerPellets, rng):
        return None if rng.random() < self.epsilon else agent


class Warden(Ghost):
    """Guards the remaining power pellet closest to the agent."""

    name, symbol = "warden", "W"

    def target(self, grid, agent, powerPellets, rng):
        if not powerPellets:
            return agent
        return min(powerPellets, key=lambda p: grid.distance(p, agent))


class Drifter(Ghost):
    """Wanders until the agent comes within `radius`, then chases."""

    name, symbol = "drifter", "D"

    def __init__(self, home: Pos, releaseDelay: int = 0, radius: int = 5):
        super().__init__(home, releaseDelay)
        self.radius = radius

    def target(self, grid, agent, powerPellets, rng):
        return agent if grid.distance(self.pos, agent) <= self.radius else None


GHOST_TYPES = (Hound, Warden, Drifter)
