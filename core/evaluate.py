import os
import gc
import multiprocessing as mp
import numpy as np

from core.dynamics import rollout, rms
from core.road_profiles import iso_random_profile


def eval_sac_on_road(model, zr_arr, cfg):
    """Evaluate a trained SAC model on a specific road profile."""
    passive_policy = lambda state, k: cfg['c0']
    yp_e, x1dd_p, _ = rollout(zr_arr, passive_policy, cfg)
    eps = 1e-6
    a_ref    = max(rms(x1dd_p),              eps)
    s_ref_ep = max(rms(yp_e[0] - yp_e[2]),  eps)
    d_ref_ep = max(rms(yp_e[2] - zr_arr),   eps)
    ctx = [0.]

    def policy(state, k):
        x1, x1d, x2, x2d = state
        zr = zr_arr[min(k, len(zr_arr)-1)]
        obs = np.array([
            (x1-x2)/s_ref_ep,
            (x1d-x2d)/cfg['V_REF'],
            (x2-zr)/d_ref_ep,
            x1d/cfg['V_REF'],
            x2d/cfg['V_REF'],
            ctx[0],
        ], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1., 1.))
        ctx[0] = a
        return cfg['C_MIN'] + (a + 1.) / 2. * (cfg['C_MAX'] - cfg['C_MIN'])

    return rollout(zr_arr, policy, cfg)


def _robustness_worker(args):
    """Top-level worker for multiprocessing — loads its own model copy."""
    model_path, cfg, Gq0, v, seed = args
    import numpy as np
    from stable_baselines3 import SAC
    from core.dynamics import rollout, rms
    from core.road_profiles import iso_random_profile
    from core.evaluate import eval_sac_on_road

    model = SAC.load(model_path, device='cpu')

    t_sweep = np.arange(int(cfg['T_SIM'] / cfg['DT'])) * cfg['DT']
    zr      = iso_random_profile(t_sweep, v=v, Gq0=Gq0, seed=seed)

    passive_policy = lambda state, k: cfg['c0']
    _, pa_sw, _ = rollout(zr, passive_policy, cfg)
    _, sa_sw, _ = eval_sac_on_road(model, zr, cfg)

    del model
    gc.collect()
    return rms(pa_sw) / max(rms(sa_sw), 1e-9)


def robustness_sweep(model, cfg, tmp_dir, n_workers=4):
    """
    Parallel robustness sweep: ISO road classes A-D x speeds [5, 10, 20] m/s.
    Returns imp_grid (shape 4x3), road_classes dict, speeds list.
    n_workers capped at number of tasks to avoid spawning idle processes.
    """
    road_classes = {'A': 16e-6, 'B': 64e-6, 'C': 256e-6, 'D': 1024e-6}
    speeds       = [5, 10, 20]

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, '_sweep_model')
    model.save(tmp_path)

    tasks = [
        (tmp_path + '.zip', cfg, Gq0, v, 7)
        for Gq0 in road_classes.values()
        for v  in speeds
    ]
    n_proc = min(n_workers, len(tasks))

    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_proc) as pool:
            results = pool.map(_robustness_worker, tasks)
    finally:
        tmp_file = tmp_path + '.zip'
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    imp_grid = np.array(results).reshape(len(road_classes), len(speeds))
    return imp_grid, road_classes, speeds


def compute_metrics_table(controllers, zr_arr, cfg):
    """
    Returns a dict of {name: {body_accel, susp_travel, tyre_defl}} for each controller.
    controllers: list of (name, y_arr, accel_arr)
    """
    metrics = {}
    for name, y, acc in controllers:
        metrics[name] = {
            'body_accel':  rms(acc),
            'susp_travel': rms(y[0] - y[2]) * 100,
            'tyre_defl':   rms(y[2] - zr_arr) * 100,
        }
    return metrics
