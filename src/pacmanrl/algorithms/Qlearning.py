"""Tabular Q-learning.
    agent = QLearning(nActions=env.action_space.n, seed=0)
    action = agent.act(obs)
    agent.update(obs, action, reward, nextObs, terminated)
    agent.decayEpsilon()  # once per episode
"""

import pickle
from collections import defaultdict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

import numpy as np


def defaultKey(state: Any) -> Hashable:
    """Turn an observation into a hashable Q-table key."""
    if isinstance(state, np.ndarray):
        return state.item() if state.ndim == 0 else tuple(state.ravel().tolist())
    if isinstance(state, np.generic):
        return state.item()
    if isinstance(state, (list, tuple)):
        return tuple(defaultKey(s) for s in state)
    if isinstance(state, dict):
        return tuple(sorted((k, defaultKey(v)) for k, v in state.items()))
    return state


class QLearning:
    def __init__(
        self,
        nActions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilonMin: float = 0.05,
        epsilonDecay: float = 0.9997,
        initialQ: float = 0.0,
        keyFn: Callable[[Any], Hashable] = defaultKey,
        seed: int | None = None,
    ):
        """
        nActions:     size of the discrete action space
        alpha:        learning rate, how far each update moves Q(s, a)
        gamma:        discount factor for future rewards
        epsilon:      initial exploration probability
        epsilonMin:   floor that `decayEpsilon` never goes below
        epsilonDecay: multiplicative decay applied by `decayEpsilon`
        initialQ:     value of unseen state-actions (> 0 = optimistic, explores more)
        keyFn:        maps an observation to a hashable Q-table key
        """
        self.nActions = nActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.initialQ = initialQ
        self.keyFn = keyFn
        self.rng = np.random.default_rng(seed)
        self.q: defaultdict[Hashable, np.ndarray] = defaultdict(self.newRow)

    def newRow(self) -> np.ndarray:
        return np.full(self.nActions, self.initialQ, dtype=np.float64)

    def values(self, state: Any) -> np.ndarray:
        """Q-values of every action in `state`."""
        return self.q[self.keyFn(state)]

    def bestAction(self, state: Any) -> int:
        """Greedy action, breaking ties randomly (plain argmax always picks action 0)."""
        values = self.values(state)
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def act(self, state: Any, greedy: bool = False) -> int:
        """Epsilon-greedy action; `greedy=True` disables exploration (for evaluation)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nActions))
        return self.bestAction(state)

    def update(
        self, state: Any, action: int, reward: float, nextState: Any, terminated: bool
    ) -> float:
        """One Q-learning step; returns the TD error.

        Pass `terminated`, not `terminated or truncated`: a time-limit cut-off
        still has future value, a real terminal state does not.
        """
        values = self.values(state)
        target = reward if terminated else reward + self.gamma * self.values(nextState).max()
        tdError = target - values[action]
        values[action] += self.alpha * tdError
        return float(tdError)

    def decayEpsilon(self) -> None:
        """Call once per episode."""
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

    def save(self, path: str | Path) -> None:
        """Pickle the Q-table and hyperparameters (not `keyFn` or the RNG)."""
        data = {
            "q": dict(self.q),
            "params": {
                "nActions": self.nActions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilonMin": self.epsilonMin,
                "epsilonDecay": self.epsilonDecay,
                "initialQ": self.initialQ,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls,
        path: str | Path,
        keyFn: Callable[[Any], Hashable] = defaultKey,
        seed: int | None = None,
    ) -> "QLearning":
        """Load a saved agent. Pass the same `keyFn` it was trained with."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(**data["params"], keyFn=keyFn, seed=seed)
        agent.q.update(data["q"])
        return agent
