import numpy as np
from scipy.integrate import solve_ivp


def rms(a):
    return np.sqrt(np.mean(a**2))


def step_sim(state, c, zr, cfg):
    ms, mu, ks, kt, DT = cfg['ms'], cfg['mu'], cfg['ks'], cfg['kt'], cfg['DT']
    x1, x1d, x2, x2d = state
    x1dd = (-ks*(x1-x2) - c*(x1d-x2d)) / ms
    x2dd = ( ks*(x1-x2) + c*(x1d-x2d) - kt*(x2-zr)) / mu
    x1d_n = x1d + x1dd*DT
    x2d_n = x2d + x2dd*DT
    x1_n  = x1  + x1d_n*DT
    x2_n  = x2  + x2d_n*DT
    return (x1_n, x1d_n, x2_n, x2d_n), x1dd, x2dd


def rollout(zr_arr, policy_fn, cfg):
    N_SUB = cfg['N_SUB']
    n = len(zr_arr)
    y      = np.empty((4, n))
    x1dd_a = np.empty(n)
    c_a    = np.empty(n)
    state  = (0., 0., 0., 0.)
    c = policy_fn(state, 0)
    for k in range(n):
        if k % N_SUB == 0:
            c = policy_fn(state, k)
        y[0,k]=state[0]; y[1,k]=state[1]; y[2,k]=state[2]; y[3,k]=state[3]
        c_a[k] = c
        state, x1dd, _ = step_sim(state, c, zr_arr[k], cfg)
        x1dd_a[k] = x1dd
    return y, x1dd_a, c_a


def simulate_passive(zr_arr, t_arr, cfg):
    ms, mu, ks, c0, kt, DT = (
        cfg['ms'], cfg['mu'], cfg['ks'], cfg['c0'], cfg['kt'], cfg['DT']
    )
    def ode_rhs(t, y):
        x1, x1d, x2, x2d = y
        zr = float(np.interp(t, t_arr, zr_arr))
        x1dd = (-ks*(x1-x2) - c0*(x1d-x2d)) / ms
        x2dd = ( ks*(x1-x2) + c0*(x1d-x2d) - kt*(x2-zr)) / mu
        return [x1d, x1dd, x2d, x2dd]
    sol = solve_ivp(ode_rhs, [0, t_arr[-1]], [0,0,0,0],
                    t_eval=t_arr, method='RK45', max_step=DT)
    return sol.t, sol.y
