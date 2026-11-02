"""Static maze structure: moves, tunnels and precomputed path distances."""

from collections import deque
import numpy as np
from .layout import Layout, Pos

UP, DOWN, LEFT, RIGHT = range(4)
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = (DOWN, UP, RIGHT, LEFT)
UNREACHABLE = 10_000


class Grid:
    def __init__(self, layout: Layout):
        self.walls = layout.walls
        self.height, self.width = layout.shape

        self.partner: dict[Pos, Pos] = {}
        for a, b in layout.tunnels:
            self.partner[a], self.partner[b] = b, a

        # list of all walkable tiles in grid
        self.cells: list[Pos] = [
            (int(r), int(c)) for r, c in zip(*np.nonzero(~self.walls))
        ]
        self.index = {pos: i for i, pos in enumerate(self.cells)}

        # next[i, a] = cell index reached by taking action a from cell i, or -1.
        self.next = np.full((len(self.cells), 4), -1, dtype=np.int64)
        for i, pos in enumerate(self.cells):
            for a in range(4):
                target = self.rawStep(pos, a)
                if target is not None:
                    self.next[i, a] = self.index[target]

        self.dist = self.allPairsBFS()

        unreachable = [
            pos
            for pos in zip(*np.nonzero(layout.pellets | layout.powerPellets))
            if self.distance(layout.agentStart, (int(pos[0]), int(pos[1])))
            >= UNREACHABLE
        ]
        if unreachable:
            raise ValueError(f"pellets unreachable from start: {unreachable}")

    def rawStep(self, pos: Pos, action: int) -> Pos | None:
        r, c = pos[0] + DELTAS[action][0], pos[1] + DELTAS[action][1]
        if not (0 <= r < self.height and 0 <= c < self.width):
            return self.partner.get(pos)
        return None if self.walls[r, c] else (r, c)

    def allPairsBFS(self) -> np.ndarray:
        n = len(self.cells)
        dist = np.full((n, n), UNREACHABLE, dtype=np.int64)
        for src in range(n):
            dist[src, src] = 0
            queue = deque([src])
            while queue:
                i = queue.popleft()
                for j in self.next[i]:
                    if j >= 0 and dist[src, j] == UNREACHABLE:
                        dist[src, j] = dist[src, i] + 1
                        queue.append(j)
        return dist

    def step(self, pos: Pos, action: int) -> Pos | None:
        """Cell reached by taking `action` from `pos`, or None if blocked."""
        j = self.next[self.index[pos], action]
        return self.cells[j] if j >= 0 else None

    def move(self, pos: Pos, action: int) -> Pos:
        """Like `step`, but bumping into a wall leaves you where you are."""
        return self.step(pos, action) or pos

    def neighbors(self, pos: Pos) -> list[tuple[int, Pos]]:
        return [
            (a, self.cells[j]) for a, j in enumerate(self.next[self.index[pos]]) if j >= 0
        ]

    def distance(self, a: Pos, b: Pos) -> int:
        return int(self.dist[self.index[a], self.index[b]])
