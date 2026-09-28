"""
Training entry point for the course project.
Run from the project root directory:
  python course_project/train.py
  python course_project/train.py --envs 4 --steps 3000000 --warmstart
"""
import sys
import os
import argparse

# ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from course_project.config import CFG
from core.train_sac import train
from core.visualize import plot_training_curve


def main():
    parser = argparse.ArgumentParser(description='Train SAC for quarter-car suspension')
    parser.add_argument('--envs',       type=int,   default=4,
                        help='Number of parallel training environments (default 4)')
    parser.add_argument('--steps',      type=int,   default=3_000_000,
                        help='Total training timesteps (default 3M)')
    parser.add_argument('--warmstart',  action='store_true',
                        help='Continue from existing best_model.zip')
    parser.add_argument('--warmstart-path', default=None,
                        help='Explicit path to warmstart checkpoint (overrides auto-search)')
    parser.add_argument('--checkpoint-dir', default='course_project/checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--results-dir',    default='course_project/results',
                        help='Directory for saved plots')
    args = parser.parse_args()

    warmstart_path = None
    if args.warmstart_path:
        warmstart_path = args.warmstart_path
        if not os.path.exists(warmstart_path):
            print(f"Warning: --warmstart-path {warmstart_path} not found. Training from scratch.")
            warmstart_path = None
    elif args.warmstart:
        candidates = [
            os.path.join(args.checkpoint_dir, 'best', 'best_model.zip'),
            'course_project/checkpoints/best/best_model.zip',     # phase-1 best
            'course_project/checkpoints_v3/best/best_model.zip',  # legacy
        ]
        for p in candidates:
            if os.path.exists(p):
                warmstart_path = p
                break
        if warmstart_path is None:
            print("Warning: --warmstart set but no checkpoint found. Training from scratch.")

    print(f"Training SAC  |  envs={args.envs}  steps={args.steps:,}")
    if warmstart_path:
        print(f"Warmstart: {warmstart_path}")

    model, ep_rewards = train(
        cfg=CFG,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        n_envs=args.envs,
        total_timesteps=args.steps,
        warmstart_path=warmstart_path,
        seed=42,
    )

    # Save training curve
    if ep_rewards:
        fig = plot_training_curve(ep_rewards, CFG, results_dir=args.results_dir)
        plt.close(fig)
        print(f"Training curve saved to {args.results_dir}/")
    else:
        print("No episode rewards logged (warmstart with no new episodes?)")

    print("\nDone. Run  python course_project/eval.py  to generate all evaluation plots.")


if __name__ == '__main__':
    main()
