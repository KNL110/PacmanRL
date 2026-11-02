"""Load experiment settings from a YAML file into typed dataclasses.

    cfg = Config.load()                    # <project root>/config.yaml
    cfg = Config.load("other.yaml")
    env = gym.make(cfg.env.id, **cfg.env.kwargs)
    agent = QLearning(env.action_space.n, **cfg.agent.params, seed=cfg.training.seed)

`env.kwargs` and `agent.params` are free-form dicts so the same config shape
works for any env or agent; `training` is fixed and checked for typos.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class EnvConfig:
    id: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 30_000
    seed: int | None = 0
    logEvery: int = 500
    evalEpisodes: int = 300
    savePath: str = "checkpoints/qlearning.pkl"

    @property
    def savePathAbs(self) -> Path:
        """`savePath`, resolved against the project root if relative."""
        path = Path(self.savePath)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Config:
    env: EnvConfig
    agent: AgentConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.fromDict(raw)

    @classmethod
    def fromDict(cls, raw: dict[str, Any]) -> "Config":
        checkKeys(raw, cls, "config")
        for section in ("env", "agent"):
            if section not in raw:
                raise ValueError(f"config is missing the '{section}' section")
        return cls(
            env=build(EnvConfig, raw["env"], "env"),
            agent=build(AgentConfig, raw["agent"], "agent"),
            training=build(TrainingConfig, raw.get("training") or {}, "training"),
        )


def checkKeys(raw: dict[str, Any], cls: type, where: str) -> None:
    """Reject unknown keys, so a typo like `epsiodes` fails loudly."""
    if not isinstance(raw, dict):
        raise ValueError(f"'{where}' must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(unknown)}; expected {sorted(known)}")


def build[T](cls: type[T], raw: dict[str, Any], where: str) -> T:
    checkKeys(raw, cls, where)
    try:
        return cls(**raw)
    except TypeError as e:  # e.g. a required key like env.id is missing
        raise ValueError(f"invalid '{where}' section: {e}") from None
