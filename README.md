# Reinforcement Learning for Semi-Active Vehicle Suspension

A simulation study applying Soft Actor-Critic (SAC) reinforcement learning to a 2-DOF quarter-car semi-active suspension model. The agent learns to vary damping continuously within a physically realisable dissipative range, and is evaluated against passive, skyhook, and LQR-semi baselines across bump and ISO 8608 random road inputs.

Physics cross-validated against an independent Scilab/Xcos implementation to four significant figures.

## Results Summary

| Metric | Passive | Skyhook | LQR-semi | SAC |
|--------|---------|---------|----------|-----|
| Body Accel RMS, Bump (m/s²) | 1.3006 | 1.7939 | 1.0096 | **1.2470** |
| Body Accel RMS, Random (m/s²) | 0.5103 | 0.5878 | 0.3973 | 0.5320 |
| Suspension Travel RMS, Random (cm) | 0.4070 | 0.4476 | 0.4376 | **0.3573** |
| Tyre Load / Static (Random) | 0.1139 | 0.1261 | 0.1598 | **0.1145** |

Road-holding constraint (tyre load ≤ 1.1× passive = 0.1253): SAC satisfies it; LQR-semi violates it by 40%.

All result plots are saved as PDF and PNG in `course_project/results/`.

## Repository Structure

```
core/
    dynamics.py          # Semi-implicit Euler stepper, rollout, RMS
    road_profiles.py     # Haversine bump and ISO 8608 random generator
    env.py               # Gymnasium environment (QuarterCarSemiEnv)
    baseline.py          # Passive, skyhook, and LQR-semi policies
    wk_filter.py         # ISO 2631-1 Wk comfort weighting filter
    evaluate.py          # Evaluation functions and parallel robustness sweep
    visualize.py         # Paper-quality plotting functions (saves PDF + PNG)
    train_sac.py         # SAC training with SubprocVecEnv parallel environments

course_project/
    config.py            # Physical parameters and training configuration (CFG dict)
    train.py             # Training entry point (CLI)
    eval.py              # Evaluation and plot generation entry point (CLI)
    notebook.ipynb       # Development notebook
    results/             # Generated figures (13 plots, PDF + PNG)

scilab/
    Quarter_car_passive.zcos    # Xcos block diagram (passive validation)
    Plots/                      # Xcos simulation output screenshots
```

## Setup

```bash
cd course_project
python -m venv .venv
source .venv/bin/activate
pip install "stable-baselines3[extra]" gymnasium numpy scipy matplotlib
```

## Training

```bash
# Fresh training (3M steps, 4 parallel environments)
python course_project/train.py --envs 4 --steps 3000000

# Continue from an existing checkpoint
python course_project/train.py --envs 4 --steps 2000000 \
    --warmstart-path course_project/checkpoints/best/best_model.zip \
    --checkpoint-dir course_project/checkpoints_phase2
```

Checkpoints are saved to `--checkpoint-dir`. The best model (by validation reward) is written to `<checkpoint-dir>/best/best_model.zip`.

Trained weights are not included in this repository due to size. Running training from scratch reproduces the reported results.

## Evaluation

```bash
python course_project/eval.py \
    --checkpoint course_project/checkpoints_phase2/best/best_model.zip \
    --results-dir course_project/results
```

This runs all four controllers on bump and random roads, prints the full metrics table, runs the parallel robustness sweep (4 road classes × 3 speeds), and saves all 13 figures.

## Physical Model

Two-degree-of-freedom quarter-car:

```
ms * x1'' = -ks*(x1-x2) - c*(x1'-x2')
mu * x2'' =  ks*(x1-x2) + c*(x1'-x2') - kt*(x2-zr)
```

The semi-active damping coefficient c is constrained to [200, 4000] Ns/m (dissipative only). The RL agent outputs a normalised action a in [-1, 1] mapped linearly to this range.

Integration uses a semi-implicit (symplectic) Euler stepper at 1 kHz with 100 Hz control. Validated against scipy RK45 with under 1% RMS deviation.

## Xcos Validation

An independent Scilab/Xcos block diagram (`scilab/Quarter_car_passive.zcos`) implements the same equations. Passive simulation on the haversine bump:

| Metric | Python | Xcos | Difference |
|--------|--------|------|-----------|
| Body Accel RMS (m/s²) | 1.2943 | 1.2945 | 0.015% |
| Suspension Travel RMS (cm) | 0.6849 | 0.6850 | 0.015% |

## Dependencies

- Python 3.11
- stable-baselines3 >= 2.0
- gymnasium
- numpy, scipy, matplotlib
- torch (CPU)
