# PACMAN RL Agent

An RL agent for a Pacman maze game.(currently it has only QLEarning implemented)

## Usage

```
uv run pacmanrl --train      # train, save to checkpoints/, then evaluate
uv run pacmanrl --evaluate   # evaluate the saved agent
uv run pacmanrl --watch      # watch the saved agent play
```

With no flag it uses `run.mode` from `config.yaml`. All settings (rewards, hyperparameters, episodes) live in `config.yaml`.

To log training to a file:

```
uv run pacmanrl --train >> train.logs
```
