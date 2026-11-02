"""Train, evaluate and watch agents, driven by config.yaml.

    uv run pacmanrl              # does whatever `run.mode` in config.yaml says
    uv run pacmanrl --train      # or --watch / --evaluate, overriding run.mode

    from pacmanrl.main import evaluate, loadAgent, watch
    cfg = Config.load()
    watch(cfg, loadAgent(cfg), nEpisodes=3)

Nothing here is specific to MazeChase: any env with a discrete action space
and any agent registered in AGENTS works.
"""

import argparse
import sys
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pacmanrl.environment  # noqa: F401  (registers MazeChase-v0)
from pacmanrl.algorithms import QLearning
from pacmanrl.config import Config

class Agent(Protocol):
    """What the training loop needs from an agent:
    any class with these members fits (QLearning already does)."""

    epsilon: float

    def act(self, state: Any, greedy: bool = False) -> int: ...
    def update(
        self, state: Any, action: int, reward: float, nextState: Any, terminated: bool
    ) -> float: ...
    def decayEpsilon(self) -> None: ...
    def save(self, path: str | Path) -> None: ...


class AgentType(Protocol):
    """An agent class: constructed as cls(nActions, **params, seed=...), loadable from disk.
    `*args: Any, **kwargs: Any` lets each agent take its own hyperparameters."""

    def __call__(self, *args: Any, **kwargs: Any) -> Agent: ...
    def load(self, path: str | Path, *, seed: int | None = None) -> Agent: ...


# Agent name in config.yaml -> class.
AGENTS: dict[str, AgentType] = {"QLearning": QLearning}

# Info keys that duplicate the return/length columns already logged.
SKIP_INFO_KEYS = {"score", "steps"}


def makeEnv(cfg: Config, renderMode: str | None = None) -> gym.Env:
    return gym.make(cfg.env.id, render_mode=renderMode, **cfg.env.kwargs)


def agentClass(cfg: Config) -> AgentType:
    if cfg.agent.name not in AGENTS:
        raise ValueError(f"unknown agent {cfg.agent.name!r}; available: {sorted(AGENTS)}")
    return AGENTS[cfg.agent.name]


def makeAgent(cfg: Config, env: gym.Env) -> Agent:
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError(f"{cfg.agent.name} needs a Discrete action space, got {env.action_space}")
    return agentClass(cfg)(int(env.action_space.n), **cfg.agent.params, seed=cfg.training.seed)


def loadAgent(cfg: Config) -> Agent:
    path = cfg.training.savePathAbs
    if not path.exists():
        raise FileNotFoundError(f"no saved agent at {path}; run training first")
    return agentClass(cfg).load(path, seed=cfg.training.seed)


def runEpisode(
    env: gym.Env,
    agent: Agent,
    learn: bool,
    seed: int | None = None,
    onStep: Callable[[int, float], None] | None = None,
) -> tuple[float, int, dict[str, Any]]:
    """Play one episode. With `learn`, the agent explores and updates its values.
    `onStep(action, reward)` is called after every step, e.g. to pause while watching."""
    obs, info = env.reset(seed=seed)
    totalReward, length, done = 0.0, 0, False
    while not done:
        action = agent.act(obs, greedy=not learn)
        nextObs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)  # Gymnasium types it as SupportsFloat
        if learn:
            agent.update(obs, action, reward, nextObs, terminated)
        obs = nextObs
        totalReward += reward
        length += 1
        done = terminated or truncated
        if onStep is not None:
            onStep(action, reward)
    return totalReward, length, info


def summarize(episodes: list[tuple[float, int, dict[str, Any]]]) -> str:
    """One log line: mean return/length, mean numeric info values, outcome counts."""
    returns = [r for r, _, _ in episodes]
    lengths = [n for _, n, _ in episodes]
    parts = [
        f"return {np.mean(returns):8.2f} ± {np.std(returns):6.2f}",
        f"length {np.mean(lengths):6.1f}",
    ]
    infos = [info for _, _, info in episodes]
    for key, value in infos[-1].items():
        if key not in SKIP_INFO_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key} {np.mean([i[key] for i in infos]):.1f}")
    outcomes = Counter(i["outcome"] for i in infos if "outcome" in i)
    if outcomes:
        parts.append("outcomes " + " ".join(f"{k}={v}" for k, v in sorted(outcomes.items(), key=str)))
    return "  ".join(parts)


def evaluate(cfg: Config, agent: Agent, nEpisodes: int) -> None:
    """Greedy episodes, no learning, on seeds different from training."""
    env = makeEnv(cfg)
    seed = None if cfg.training.seed is None else cfg.training.seed + 1
    episodes = [runEpisode(env, agent, learn=False, seed=seed if i == 0 else None) for i in range(nEpisodes)]
    env.close()
    print(f"eval ({nEpisodes} greedy episodes): {summarize(episodes)}")


def train(cfg: Config) -> Agent:
    """Train, save, evaluate; returns the trained agent."""
    t = cfg.training
    env = makeEnv(cfg)
    agent = makeAgent(cfg, env)
    print(f"training {cfg.agent.name} on {cfg.env.id} for {t.episodes} episodes")

    window: deque[tuple[float, int, dict[str, Any]]] = deque(maxlen=t.logEvery)
    for ep in range(1, t.episodes + 1):
        # Seed only the first reset; later resets continue the same RNG stream.
        window.append(runEpisode(env, agent, learn=True, seed=t.seed if ep == 1 else None))
        agent.decayEpsilon()
        if ep % t.logEvery == 0 or ep == t.episodes:
            table = getattr(agent, "q", None)  # tabular agents only
            extra = f"  states {len(table)}" if table is not None else ""
            print(f"ep {ep:>7}  {summarize(list(window))}  eps {agent.epsilon:.3f}{extra}")
    env.close()

    path = t.savePathAbs
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    print(f"saved agent to {path}")

    if t.evalEpisodes > 0:
        evaluate(cfg, agent, t.evalEpisodes)
    return agent


def watch(cfg: Config, agent: Agent, nEpisodes: int = 1) -> None:
    """Play greedy episodes on screen at `run.renderFps`; with `run.stepByStep`,
    show each action and reward and wait for Enter."""
    env = makeEnv(cfg, renderMode="human")
    # Per-instance copy, so the class-level default is left untouched.
    env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": cfg.run.renderFps}
    names = getattr(env.unwrapped, "actionNames", None)

    def pause(action: int, reward: float) -> None:
        label = names[action] if names else str(action)
        input(f"action {label:<6} reward {reward:+.2f}   [Enter = next step, Ctrl+C = quit] ")

    try:
        for i in range(nEpisodes):
            totalReward, length, info = runEpisode(
                env, agent, learn=False, onStep=pause if cfg.run.stepByStep else None
            )
            print(f"episode {i + 1}: return {totalReward:.2f}  length {length}  {info.get('outcome', '')}")
    except (KeyboardInterrupt, EOFError):  # Ctrl+C, or Ctrl+D at the prompt
        print("\nstopped")
    finally:
        env.close()


def parseMode() -> str | None:
    """The mode flag, if given: --train, --watch or --evaluate (-train etc. also work)."""
    parser = argparse.ArgumentParser(prog="pacmanrl", description="Train, evaluate or watch an agent.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", "-train", dest="mode", action="store_const", const="train")
    group.add_argument("--watch", "-watch", dest="mode", action="store_const", const="watch")
    group.add_argument("--evaluate", "-evaluate", dest="mode", action="store_const", const="eval")
    return parser.parse_args().mode


def main() -> None:
    # Flush each line even when stdout is redirected (e.g. `>> train.logs`).
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    cfg = Config.load()
    if (mode := parseMode()) is not None:
        cfg = replace(cfg, run=replace(cfg.run, mode=mode))
    match cfg.run.mode:
        case "train":
            train(cfg)
        case "eval":
            evaluate(cfg, loadAgent(cfg), cfg.training.evalEpisodes)
        case "watch":
            watch(cfg, loadAgent(cfg), cfg.run.watchEpisodes)


if __name__ == "__main__":
    main()
