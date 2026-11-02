"""ASCII maze layouts and parsing.

Symbols:
    #  wall
    .  pellet
    o  power pellet
    P  agent start
    G  ghost spawn (one per ghost, in order of appearance)
    T  tunnel end; tunnel ends pair up in order of appearance
       and wrap to each other when walked off the edge
    (space) empty floor
"""

from dataclasses import dataclass
import numpy as np
Pos = tuple[int, int]

DEFAULT_LAYOUT = """\
###################
#o.......#.......o#
#.##.###.#.###.##.#
#.................#
#.##.#.#####.#.##.#
#....#...G...#....#
####.###.#.###.####
T.......G.G.......T
####.#.#####.#.####
#........P........#
#.##.###.#.###.##.#
#o..#.........#..o#
###################
"""


@dataclass(frozen=True)
class Layout:
    walls: np.ndarray  # bool (H, W)
    pellets: np.ndarray  # bool (H, W)
    powerPellets: np.ndarray  # bool (H, W)
    agentStart: Pos
    ghostStarts: tuple[Pos, ...]
    tunnels: tuple[tuple[Pos, Pos], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape


def parseLayout(text: str) -> Layout:
    rows = [line for line in text.splitlines() if line.strip()]
    height, width = len(rows), max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]

    walls = np.zeros((height, width), dtype=bool)
    pellets = np.zeros_like(walls)
    power = np.zeros_like(walls)
    agentStart: Pos | None = None
    ghostStarts: list[Pos] = []
    tunnelEnds: list[Pos] = []

    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            match ch:
                case "#":
                    walls[r, c] = True
                case ".":
                    pellets[r, c] = True
                case "o":
                    power[r, c] = True
                case "P":
                    if agentStart is not None:
                        raise ValueError("layout has more than one 'P'")
                    agentStart = (r, c)
                case "G":
                    ghostStarts.append((r, c))
                case "T":
                    tunnelEnds.append((r, c))
                case " ":
                    pass
                case _:
                    raise ValueError(f"unknown layout symbol {ch!r} at {(r, c)}")

    if agentStart is None:
        raise ValueError("layout has no 'P'")
    if len(tunnelEnds) % 2:
        raise ValueError("tunnel ends 'T' must come in pairs")

    # Only tunnel ends may sit on the border without a wall.
    border = np.ones_like(walls)
    border[1:-1, 1:-1] = False
    for r, c in zip(*np.nonzero(border & ~walls)):
        if (r, c) not in tunnelEnds:
            raise ValueError(f"open border tile {(int(r), int(c))} is not a tunnel")

    tunnels = tuple(zip(tunnelEnds[::2], tunnelEnds[1::2]))
    return Layout(walls, pellets, power, agentStart, tuple(ghostStarts), tunnels)
