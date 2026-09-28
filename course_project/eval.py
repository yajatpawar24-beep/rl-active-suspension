"""
Evaluation and plotting entry point for the course project.
Run from the project root directory:
  python course_project/eval.py
  python course_project/eval.py --checkpoint path/to/best_model.zip
  python course_project/eval.py --no-sweep   # skip robustness sweep (faster)
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from course_project.config import CFG
from core.dynamics import rollout, rms
from core.road_profiles import bump_profile, iso_random_profile
from core.baseline import make_passive_policy, make_skyhook_policy, make_lqr_semi_policy
from core.wk_filter import make_wk_sos, wk_rms
from core.evaluate import eval_sac_on_road, robustness_sweep, compute_metrics_table
from core import visualize as viz

from stable_baselines3 import SAC


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model: {checkpoint_path}")
    return SAC.load(checkpoint_path, device='cpu')


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC and generate all plots')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model .zip (default: search checkpoints/best/best_model.zip)')
    parser.add_argument('--results-dir',  default='course_project/results',
                        help='Output directory for plots')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Skip robustness sweep (runs 12 parallel sims)')
    parser.add_argument('--sweep-workers', type=int, default=4,
                        help='Parallel workers for robustness sweep (default 4)')
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Find model
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        candidates = [
            'course_project/checkpoints/best/best_model.zip',
            'course_project/checkpoints_v3/best/best_model.zip',
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Run  python course_project/train.py  first."
            )

    model = load_model(model_path)

    # Road profiles
    t_arr   = np.arange(0, CFG['T_SIM'], CFG['DT'])
    zr_bump = bump_profile(t_arr)
    zr_rand = iso_random_profile(t_arr)

    # Baseline controllers
    passive_policy = make_passive_policy(CFG)
    skyhook_policy = make_skyhook_policy(c_sky=2500.0, cfg=CFG)
    lqr_policy     = make_lqr_semi_policy(CFG)

    print("Running baselines...")
    yp_b,  pa_b, _       = rollout(zr_bump, passive_policy, CFG)
    yp_r,  pa_r, _       = rollout(zr_rand, passive_policy, CFG)
    ysk_b, sk_b, _       = rollout(zr_bump, skyhook_policy, CFG)
    ysk_r, sk_r, _       = rollout(zr_rand, skyhook_policy, CFG)
    ylq_b, lq_b, _       = rollout(zr_bump, lqr_policy,     CFG)
    ylq_r, lq_r, _       = rollout(zr_rand, lqr_policy,     CFG)

    print("Evaluating SAC...")
    ysa_b, sa_b, sac_c_b = eval_sac_on_road(model, zr_bump, CFG)
    ysa_r, sa_r, sac_c_r = eval_sac_on_road(model, zr_rand, CFG)

    # Print summary table
    print("\n" + "="*68)
    print(f"{'Metric':<32} {'Passive':>8} {'Skyhook':>8} {'LQR-semi':>9} {'SAC':>8}")
    print("="*68)
    for lbl, pa_, sk_, lq_, sa_ in [
        ('Body Accel RMS (m/s2) - Bump',   pa_b, sk_b, lq_b, sa_b),
        ('Body Accel RMS (m/s2) - Random', pa_r, sk_r, lq_r, sa_r),
    ]:
        print(f"  {lbl:<30} {rms(pa_):>8.4f} {rms(sk_):>8.4f} {rms(lq_):>9.4f} {rms(sa_):>8.4f}")
    print("-"*68)
    for lbl, yp_, ysk_, ylq_, ysa_, zr_, is_susp in [
        ('Susp Travel RMS (cm) - Bump',   yp_b, ysk_b, ylq_b, ysa_b, zr_bump, True),
        ('Susp Travel RMS (cm) - Random', yp_r, ysk_r, ylq_r, ysa_r, zr_rand, True),
        ('Tyre Defl RMS (cm) - Bump',     yp_b, ysk_b, ylq_b, ysa_b, zr_bump, False),
        ('Tyre Defl RMS (cm) - Random',   yp_r, ysk_r, ylq_r, ysa_r, zr_rand, False),
    ]:
        f = lambda y, z, s=is_susp: rms(y[0]-y[2] if s else y[2]-z) * 100
        print(f"  {lbl:<30} {f(yp_,zr_):>8.4f} {f(ysk_,zr_):>8.4f} {f(ylq_,zr_):>9.4f} {f(ysa_,zr_):>8.4f}")
    print("="*68)

    # Dynamic tyre load
    static = CFG['STATIC_LOAD']
    kt     = CFG['kt']
    print("\nDynamic tyre load RMS / static (<=1.1x passive = constraint):")
    for lbl, y_, zr_ in [
        ('Passive', yp_r, zr_rand), ('Skyhook', ysk_r, zr_rand),
        ('LQR-semi', ylq_r, zr_rand), ('SAC', ysa_r, zr_rand),
    ]:
        print(f"  {lbl:<10}: {rms(kt*(y_[2]-zr_))/static:.4f}")

    # Wk-weighted comfort
    WK_SOS = make_wk_sos(1.0/CFG['DT'])
    print("\nISO 2631-1 Wk-Weighted RMS - Random Road:")
    wp = wk_rms(pa_r, WK_SOS); ws = wk_rms(sk_r, WK_SOS)
    wl = wk_rms(lq_r, WK_SOS); wsa = wk_rms(sa_r, WK_SOS)
    for lbl, w_ in [('Passive', wp), ('Skyhook', ws), ('LQR-semi', wl), ('SAC', wsa)]:
        print(f"  {lbl:<10}: {w_:.4f} m/s2  ({(1-w_/wp)*100:+.1f}% vs passive)")

    # Generate all plots
    print(f"\nSaving plots to {results_dir}/")

    fig = viz.plot_road_profiles(t_arr, zr_bump, zr_rand, results_dir)
    plt.close(fig)

    fig = viz.plot_passive_baseline(t_arr, yp_b, pa_b, yp_r, pa_r, zr_bump, zr_rand, results_dir)
    plt.close(fig)

    fig = viz.plot_skyhook_comparison(t_arr, yp_b, pa_b, ysk_b, sk_b, zr_bump, results_dir)
    plt.close(fig)

    fig, fig2 = viz.plot_four_controller_comparison(
        t_arr, yp_b, pa_b, ysk_b, sk_b, ylq_b, lq_b, ysa_b, sa_b,
        sac_c_b, zr_bump, road_label='Bump', results_dir=results_dir)
    plt.close(fig); plt.close(fig2)

    fig, fig2 = viz.plot_four_controller_comparison(
        t_arr, yp_r, pa_r, ysk_r, sk_r, ylq_r, lq_r, ysa_r, sa_r,
        sac_c_r, zr_rand, road_label='Random', results_dir=results_dir)
    plt.close(fig); plt.close(fig2)

    metrics_bump = compute_metrics_table(
        [('Passive', yp_b, pa_b), ('Skyhook', ysk_b, sk_b),
         ('LQR-semi', ylq_b, lq_b), ('SAC', ysa_b, sa_b)],
        zr_bump, CFG)
    metrics_rand = compute_metrics_table(
        [('Passive', yp_r, pa_r), ('Skyhook', ysk_r, sk_r),
         ('LQR-semi', ylq_r, lq_r), ('SAC', ysa_r, sa_r)],
        zr_rand, CFG)
    viz.plot_rms_bars(metrics_bump, metrics_rand, results_dir)

    fig = viz.plot_wk_filter(WK_SOS, fs=1.0/CFG['DT'], results_dir=results_dir)
    plt.close(fig)

    accels_rand = {'Passive': pa_r, 'Skyhook': sk_r, 'LQR-semi': lq_r, 'SAC': sa_r}
    fig = viz.plot_wk_comparison(t_arr, accels_rand, WK_SOS, results_dir)
    plt.close(fig)

    # Bonus paper figures
    fig = viz.plot_psd_comparison(t_arr, accels_rand, CFG, results_dir)
    plt.close(fig)

    fig = viz.plot_damping_distribution(sac_c_b, sac_c_r, CFG, results_dir)
    plt.close(fig)

    controllers_rand = [
        ('Passive',  yp_r, pa_r),
        ('Skyhook',  ysk_r, sk_r),
        ('LQR-semi', ylq_r, lq_r),
        ('SAC',      ysa_r, sa_r),
    ]
    fig = viz.plot_tyre_load(t_arr, controllers_rand, zr_rand, CFG, results_dir)
    plt.close(fig)

    # Robustness sweep (parallel)
    if not args.no_sweep:
        print("\nRunning parallel robustness sweep (4 road classes x 3 speeds)...")
        tmp_dir = os.path.join(results_dir, '_tmp')
        imp_grid, road_classes, speeds = robustness_sweep(
            model, CFG, tmp_dir=tmp_dir, n_workers=args.sweep_workers
        )
        fig = viz.plot_robustness_heatmap(imp_grid, road_classes, speeds, results_dir)
        plt.close(fig)
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nAll plots saved to {results_dir}/")
    print("Files: fig1_road_profiles, fig2_passive_baseline, fig3_skyhook_comparison,")
    print("       fig4_4ctrl_bump, fig4_4ctrl_random, fig5_sac_damping_*,")
    print("       fig6_rms_bars_*, fig8_robustness_heatmap, fig9_wk_filter,")
    print("       fig10_wk_comparison, fig11_psd_comparison,")
    print("       fig12_damping_distribution, fig13_tyre_load")
    print("       (each as .pdf + .png)")


if __name__ == '__main__':
    main()
