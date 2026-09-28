import numpy as np
from scipy.linalg import solve_continuous_are

from core.dynamics import rollout, rms


def make_passive_policy(cfg):
    c0 = cfg['c0']
    return lambda state, k: c0


def make_skyhook_policy(c_sky=2500.0, cfg=None):
    C_MIN = cfg['C_MIN'] if cfg else 200.0
    def policy(state, k):
        _, x1d, _, x2d = state
        return c_sky if x1d*(x1d-x2d) > 0 else C_MIN
    return policy


def make_lqr_semi_policy(cfg):
    ms, mu, ks, c0, kt = cfg['ms'], cfg['mu'], cfg['ks'], cfg['c0'], cfg['kt']
    C_MIN, C_MAX = cfg['C_MIN'], cfg['C_MAX']
    W2, W3, S_REF, D_REF = cfg['W2'], cfg['W3'], cfg['S_REF'], cfg['D_REF']

    A_sys = np.array([
        [0,      1,       0,            0      ],
        [-ks/ms, -c0/ms,  ks/ms,        c0/ms  ],
        [0,      0,       0,            1      ],
        [ks/mu,  c0/mu,  -(ks+kt)/mu,  -c0/mu ]
    ])
    B_sys = np.array([[0.],[1./ms],[0.],[-1./mu]])

    c_acc  = np.array([-ks/ms, -c0/ms, ks/ms, c0/ms])
    D      = 1./ms
    r_ctrl = 1e-8

    e_susp = np.array([1., 0., -1., 0.])
    e_tyre = np.array([0., 0.,  1., 0.])

    Q_raw   = (np.outer(c_acc, c_acc)
               + W2/S_REF**2 * np.outer(e_susp, e_susp)
               + W3/D_REF**2 * np.outer(e_tyre, e_tyre))
    N_cross = (D * c_acc).reshape(-1, 1)
    R_eff   = np.array([[D**2 + r_ctrl]])

    A_bar = A_sys - B_sys @ (N_cross.T / R_eff[0,0])
    Q_bar = Q_raw  - N_cross @ N_cross.T / R_eff[0,0] + 1e-6*np.eye(4)

    P     = solve_continuous_are(A_bar, B_sys, Q_bar, R_eff)
    K_lqr = (B_sys.T @ P + N_cross.T) / R_eff[0,0]

    def policy(state, k, v_guard=1e-4):
        x1, x1d, x2, x2d = state
        Fa_des = float((-K_lqr @ np.array([x1, x1d, x2, x2d]))[0])
        rel_v  = x1d - x2d
        if abs(rel_v) < v_guard:
            return c0
        c = c0 - Fa_des / rel_v
        return float(np.clip(c, C_MIN, C_MAX))

    return policy
