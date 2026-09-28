import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
from scipy.signal import welch, sosfilt

from core.dynamics import rms
from core.wk_filter import make_wk_sos

# Paper-quality style
COLORS = {
    'Passive':  '#2166ac',
    'Skyhook':  '#d6604d',
    'LQR-semi': '#f4a582',
    'SAC':      '#1a9850',
    'Road':     '#aaaaaa',
}
FIGSIZE_FULL  = (13, 9)
FIGSIZE_WIDE  = (12, 4)
FIGSIZE_HALF  = (7,  5)
DPI_SCREEN    = 110
DPI_PAPER     = 300

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': DPI_SCREEN,
    'savefig.dpi': DPI_PAPER,
    'savefig.bbox': 'tight',
})


def _save(fig, results_dir, name):
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, name + '.pdf'))
        fig.savefig(os.path.join(results_dir, name + '.png'))


def plot_road_profiles(t_arr, zr_bump, zr_rand, results_dir=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3))
    a1.plot(t_arr, zr_bump*100, color=COLORS['Passive'])
    a1.set(xlabel='Time (s)', ylabel='Road height (cm)', title='Bump Profile (5 cm haversine)')
    a2.plot(t_arr, zr_rand*100, color=COLORS['Skyhook'], lw=0.6)
    a2.set(xlabel='Time (s)', ylabel='Road height (cm)', title='ISO 8608 Class B Random')
    plt.tight_layout()
    _save(fig, results_dir, 'fig1_road_profiles')
    return fig


def plot_passive_baseline(t_arr, yp_bump, pa_b, yp_rand, pa_r,
                           zr_bump, zr_rand, results_dir=None):
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE_FULL, sharex='col')
    for col, (y_, zr_, acc_, title) in enumerate([
        (yp_bump, zr_bump, pa_b, 'Passive - Bump'),
        (yp_rand, zr_rand, pa_r, 'Passive - ISO Class B Random'),
    ]):
        axes[0,col].plot(t_arr, y_[0]*100, label='body x1')
        axes[0,col].plot(t_arr, y_[2]*100, label='wheel x2', ls='--')
        axes[0,col].plot(t_arr, zr_*100,   label='road zr',  ls=':', color=COLORS['Road'])
        axes[0,col].set_ylabel('Displacement (cm)'); axes[0,col].legend()
        axes[0,col].set_title(title)
        axes[1,col].plot(t_arr, acc_, color=COLORS['Passive'])
        axes[1,col].set_ylabel('Body Accel (m/s2)')
        axes[2,col].plot(t_arr, (y_[0]-y_[2])*100, color='purple')
        axes[2,col].set_ylabel('Susp. Travel (cm)'); axes[2,col].set_xlabel('Time (s)')
    plt.tight_layout()
    _save(fig, results_dir, 'fig2_passive_baseline')
    return fig


def plot_skyhook_comparison(t_arr, yp_bump, pa_b, ysk_bump, sk_b, zr_bump, results_dir=None):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t_arr, yp_bump[0]*100,  label='Passive',  color=COLORS['Passive'])
    axes[0].plot(t_arr, ysk_bump[0]*100, label='Skyhook',  color=COLORS['Skyhook'])
    axes[0].plot(t_arr, zr_bump*100,     color=COLORS['Road'], ls=':', label='Road')
    axes[0].set_ylabel('Body Disp. (cm)'); axes[0].legend()
    axes[0].set_title('Skyhook vs Passive - Bump (100 Hz ZOH, semi-implicit Euler)')
    axes[1].plot(t_arr, pa_b, label='Passive', color=COLORS['Passive'])
    axes[1].plot(t_arr, sk_b, label='Skyhook', color=COLORS['Skyhook'])
    axes[1].set_ylabel('Body Accel (m/s2)'); axes[1].legend()
    axes[2].plot(t_arr, (yp_bump[0]-yp_bump[2])*100, label='Passive', color=COLORS['Passive'])
    axes[2].plot(t_arr, (ysk_bump[0]-ysk_bump[2])*100, label='Skyhook', color=COLORS['Skyhook'])
    axes[2].set_ylabel('Susp. Travel (cm)'); axes[2].legend(); axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    _save(fig, results_dir, 'fig3_skyhook_comparison')
    return fig


def plot_four_controller_comparison(
    t_arr,
    yp, pa, ysk, sk, ylq, lq, ysa, sa,
    sac_c, zr, road_label='Bump',
    results_dir=None,
):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for lbl, y_, acc_ in [
        ('Passive',  yp,  pa),
        ('Skyhook',  ysk, sk),
        ('LQR-semi', ylq, lq),
        ('SAC',      ysa, sa),
    ]:
        axes[0].plot(t_arr, y_[0]*100,          label=lbl, color=COLORS[lbl])
        axes[1].plot(t_arr, acc_,               label=lbl, color=COLORS[lbl])
        axes[2].plot(t_arr, (y_[0]-y_[2])*100, label=lbl, color=COLORS[lbl])
    axes[0].plot(t_arr, zr*100, color=COLORS['Road'], ls=':', label='Road')
    axes[0].set_ylabel('Body Disp. (cm)')
    axes[0].legend(ncol=2)
    axes[0].set_title(f'4-Controller Comparison - {road_label}')
    axes[1].set_ylabel('Body Accel (m/s2)'); axes[1].legend(ncol=2)
    axes[2].set_ylabel('Susp. Travel (cm)');  axes[2].legend(ncol=2)
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    tag = road_label.lower().replace(' ', '_')
    _save(fig, results_dir, f'fig4_4ctrl_{tag}')

    # SAC damping history
    fig2, ax2 = plt.subplots(figsize=FIGSIZE_WIDE)
    ax2.plot(t_arr, sac_c, color=COLORS['SAC'], lw=0.8, label='SAC damping')
    ax2.axhline(1000.0, color=COLORS['Road'], ls='--', lw=0.8, label='Passive c0=1000')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('c (Ns/m)')
    ax2.set_title(f'SAC Damping History - {road_label}')
    ax2.legend()
    plt.tight_layout()
    _save(fig2, results_dir, f'fig5_sac_damping_{tag}')
    return fig, fig2


def plot_rms_bars(metrics_bump, metrics_rand, results_dir=None):
    metric_keys   = ['body_accel', 'susp_travel', 'tyre_defl']
    metric_labels = ['Body Accel\n(m/s2)', 'Susp Travel\n(cm)', 'Tyre Defl\n(cm)']
    controllers   = ['Passive', 'Skyhook', 'LQR-semi', 'SAC']

    for road_label, metrics in [('Bump', metrics_bump), ('Random', metrics_rand)]:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(metric_keys)); w = 0.2
        for i, lbl in enumerate(controllers):
            vals = [metrics[lbl][k] for k in metric_keys]
            ax.bar(x + (i-1.5)*w, vals, w, label=lbl, color=COLORS[lbl], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(metric_labels)
        ax.set_ylabel('RMS Value')
        ax.set_title(f'RMS Metrics - {road_label} Road')
        ax.legend()
        plt.tight_layout()
        tag = road_label.lower()
        _save(fig, results_dir, f'fig6_rms_bars_{tag}')
        plt.close(fig)


def plot_training_curve(ep_rewards, cfg, results_dir=None):
    rewards = np.array(ep_rewards) if len(ep_rewards) > 0 else np.array([0.])
    p2   = np.percentile(rewards, 2)
    rp   = rewards[rewards > p2]
    ep_i = np.where(rewards > p2)[0]
    win  = min(80, max(1, len(rp)//10))
    sm   = np.convolve(rp, np.ones(win)/win, mode='valid') if len(rp) >= win else rp

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ep_i, rp, alpha=0.12, color=COLORS['SAC'], lw=0.5, label='Episode reward')
    if len(sm) > 0:
        ax.plot(ep_i[win//2:win//2+len(sm)], sm, color=COLORS['SAC'], lw=2,
                label=f'{win}-ep moving avg')
    passive_score = -(1 + cfg['W2'] + cfg['W3']) * int(cfg['T_TRAIN'] / cfg['DT_CTRL'])
    ax.axhline(passive_score, color=COLORS['Road'], ls='--', lw=1,
               label=f'Passive baseline approx {passive_score:.0f}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
    ax.set_title('SAC Training Curve')
    ax.legend()
    plt.tight_layout()
    _save(fig, results_dir, 'fig7_training_curve')
    return fig


def plot_robustness_heatmap(imp_grid, road_classes, speeds, results_dir=None):
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    im = ax.imshow(imp_grid, cmap='RdYlGn', vmin=0.8, vmax=1.6, aspect='auto')
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f'{v} m/s' for v in speeds])
    ax.set_yticks(range(len(road_classes)))
    ax.set_yticklabels(list(road_classes.keys()))
    ax.set_xlabel('Vehicle Speed'); ax.set_ylabel('ISO Road Class')
    ax.set_title('Robustness: Passive RMS / SAC RMS\n(>1 = SAC better; A-C in-dist, D OOD)')
    plt.colorbar(im, ax=ax, label='Improvement ratio')
    for ri in range(imp_grid.shape[0]):
        for si in range(imp_grid.shape[1]):
            ax.text(si, ri, f'{imp_grid[ri,si]:.2f}',
                    ha='center', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, results_dir, 'fig8_robustness_heatmap')
    return fig


def plot_wk_filter(sos, fs=1000., results_dir=None):
    from scipy.signal import sosfreqz
    w_hz, h = sosfreqz(sos, worN=8192, fs=fs)
    peak_f = w_hz[np.argmax(np.abs(h))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(w_hz[1:], 20*np.log10(np.abs(h[1:])+1e-15),
                color=COLORS['Passive'], lw=2)
    ax.axvline(peak_f, color=COLORS['Skyhook'], ls='--', label=f'Peak {peak_f:.1f} Hz')
    for f_ in [4., 8.]:
        ax.axvline(f_, color=COLORS['Road'], ls=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
    ax.set_title('ISO 2631-1 Wk Filter - Magnitude Response')
    ax.legend(); ax.set_xlim([0.1, 500]); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save(fig, results_dir, 'fig9_wk_filter')
    return fig


def plot_wk_comparison(t_arr, accels, sos, results_dir=None):
    """accels: dict of {label: accel_array}"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for lbl, a_ in accels.items():
        c = COLORS.get(lbl, '#333333')
        wk_val = rms(sosfilt(sos, a_))
        axes[0].plot(t_arr, a_,               color=c, label=lbl, alpha=0.7)
        axes[1].plot(t_arr, sosfilt(sos, a_), color=c,
                     label=f'{lbl}  Wk-RMS={wk_val:.3f}')
    axes[0].set_ylabel('Raw Accel (m/s2)');           axes[0].legend(ncol=2)
    axes[0].set_title('ISO 2631-1 Wk Weighted Comparison - Random Road')
    axes[1].set_ylabel('Wk-filtered Accel (m/s2)'); axes[1].legend(ncol=2)
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    _save(fig, results_dir, 'fig10_wk_comparison')
    return fig


# Bonus paper figures

def plot_psd_comparison(t_arr, accels, cfg, results_dir=None):
    """Power spectral density of body acceleration - frequency-domain view."""
    fs = 1.0 / cfg['DT']
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, a_ in accels.items():
        f, Pxx = welch(a_, fs=fs, nperseg=min(1024, len(a_)//4))
        ax.semilogy(f, Pxx, label=lbl, color=COLORS.get(lbl, '#333333'), lw=1.5)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD [(m/s2)2/Hz]')
    ax.set_title('Power Spectral Density - Body Acceleration (Random Road)')
    ax.set_xlim([0.5, 50]); ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save(fig, results_dir, 'fig11_psd_comparison')
    return fig


def plot_damping_distribution(sac_c_bump, sac_c_rand, cfg, results_dir=None):
    """Histogram of SAC damping commands - shows learned strategy."""
    C_MIN, C_MAX = cfg['C_MIN'], cfg['C_MAX']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c_arr, title in [
        (axes[0], sac_c_bump, 'SAC Damping Distribution - Bump'),
        (axes[1], sac_c_rand, 'SAC Damping Distribution - Random'),
    ]:
        ax.hist(c_arr, bins=40, color=COLORS['SAC'], alpha=0.75, edgecolor='white')
        ax.axvline(1000.0, color=COLORS['Road'], ls='--', lw=1.5, label='Passive c0=1000')
        ax.set_xlabel('Damping c (Ns/m)'); ax.set_ylabel('Count')
        ax.set_title(title); ax.set_xlim([C_MIN-50, C_MAX+50])
        ax.legend()
    plt.tight_layout()
    _save(fig, results_dir, 'fig12_damping_distribution')
    return fig


def plot_tyre_load(t_arr, controllers, zr_arr, cfg, results_dir=None):
    """Dynamic tyre load normalised by static - road-holding metric."""
    static_load = (cfg['ms'] + cfg['mu']) * 9.81
    fig, ax = plt.subplots(figsize=(12, 4))
    for lbl, y_, _ in controllers:
        tyre_load = cfg['kt'] * (y_[2] - zr_arr) / static_load
        ax.plot(t_arr, tyre_load, label=lbl, color=COLORS.get(lbl, '#333333'), alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Dyn. Tyre Load / Static Load')
    ax.set_title('Dynamic Tyre Load - Random Road (road-holding constraint)')
    ax.legend(ncol=2)
    plt.tight_layout()
    _save(fig, results_dir, 'fig13_tyre_load')
    return fig
