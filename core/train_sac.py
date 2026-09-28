"""
Parallel SAC training for the quarter-car semi-active suspension.

Usage (from project root):
  python course_project/train.py               # 4 envs, 3M steps, fresh start
  python course_project/train.py --envs 6      # more envs
  python course_project/train.py --warmstart   # continue from existing best model
  python course_project/train.py --steps 5000000
"""

import os
import gc
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)

from core.env import QuarterCarSemiEnv


class EpisodeRewardLogger(BaseCallback):
    """Accumulates per-episode rewards for the training curve plot."""
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._buf = {}

    def _on_step(self):
        for i, (r, d) in enumerate(zip(
            self.locals['rewards'], self.locals['dones']
        )):
            self._buf[i] = self._buf.get(i, 0.0) + float(r)
            if d:
                self.ep_rewards.append(self._buf[i])
                self._buf[i] = 0.0
        return True


def _make_env(cfg, road_type, ep_len, rank, seed):
    """Returns a callable (required by SubprocVecEnv)."""
    def _init():
        env = QuarterCarSemiEnv(cfg, road_type=road_type, ep_len=ep_len)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    cfg,
    checkpoint_dir,
    results_dir,
    n_envs=4,
    total_timesteps=3_000_000,
    warmstart_path=None,
    seed=42,
):
    """
    Train SAC with parallel environments.
    Memory-safe defaults: buffer_size=200_000, n_envs<=6 recommended.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Spawn parallel training envs
    train_env = SubprocVecEnv([
        _make_env(cfg, road_type='mixed', ep_len=cfg['T_TRAIN'],
                  rank=i, seed=seed)
        for i in range(n_envs)
    ])
    train_env = VecMonitor(train_env)

    # Single eval env in its own subprocess
    eval_env = SubprocVecEnv([
        _make_env(cfg, road_type='random', ep_len=cfg['T_TRAIN'],
                  rank=0, seed=seed + 9999)
    ])
    eval_env = VecMonitor(eval_env)

    best_path = os.path.join(checkpoint_dir, 'best')
    os.makedirs(best_path, exist_ok=True)

    reward_logger = EpisodeRewardLogger()

    callbacks = CallbackList([
        reward_logger,
        EvalCallback(
            eval_env,
            best_model_save_path=best_path,
            log_path=os.path.join(checkpoint_dir, 'logs'),
            eval_freq=max(20_000 // n_envs, 1),
            n_eval_episodes=40,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(500_000 // n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix='sac_semi',
            verbose=1,
        ),
    ])

    best_model_zip = os.path.join(best_path, 'best_model.zip')

    if warmstart_path and os.path.exists(warmstart_path):
        print(f"Warmstarting from {warmstart_path}")
        # SAC.load(env=...) handles n_envs mismatch; set_env does not
        # Phase-2 settings: lower lr + gradient_steps=1 to escape plateau
        model = SAC.load(
            warmstart_path,
            env=train_env,
            device='cpu',
            custom_objects={
                'buffer_size': 200_000,
                'learning_starts': 0,
                'batch_size': 256,
                'gradient_steps': 1,
                'learning_rate': 1e-4,
                'gamma': 0.995,
            },
        )
    else:
        model = SAC(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            buffer_size=200_000,      # ~24 MB - memory-safe
            batch_size=512,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=-1,        # = n_envs: keeps 1:1 data/gradient ratio
            learning_starts=10_000,
            ent_coef='auto',
            use_sde=True,             # State-Dependent Exploration - smoother control
            sde_sample_freq=64,
            policy_kwargs=dict(
                net_arch=[256, 256],
                log_std_init=-3,      # start less exploratory, favour smooth policy
            ),
            device='cpu',
            verbose=1,
            seed=seed,
        )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=(warmstart_path is None),
        )
    finally:
        train_env.close()
        eval_env.close()
        gc.collect()

    # Load the validation-best checkpoint for evaluation
    if os.path.exists(best_model_zip):
        print(f"\nLoading best checkpoint from {best_model_zip}")
        model = SAC.load(best_model_zip, device='cpu')

    return model, reward_logger.ep_rewards
