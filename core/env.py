import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.dynamics import step_sim, rollout, rms
from core.road_profiles import bump_profile, iso_random_profile


class QuarterCarSemiEnv(gym.Env):
    """
    Parameterised semi-active quarter-car environment.
    cfg must contain all keys from course_project/config.py CFG dict.
    """
    metadata = {}

    def __init__(self, cfg, road_type='mixed', ep_len=None):
        super().__init__()
        self.cfg       = cfg
        self.road_type = road_type
        self.ep_len    = ep_len if ep_len is not None else cfg['T_TRAIN']
        self.n_ctrl    = int(round(self.ep_len / cfg['DT_CTRL']))
        self.n_phys    = self.n_ctrl * cfg['N_SUB']

        self.observation_space = spaces.Box(
            -10.*np.ones(6, dtype=np.float32),
             10.*np.ones(6, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1.], dtype=np.float32),
            np.array([ 1.], dtype=np.float32))

        self._zr       = np.zeros(self.n_phys)
        self._a_ref    = 1.0
        self._s_ref_ep = cfg['S_REF']
        self._d_ref_ep = cfg['D_REF']
        self._a_prev   = 0.
        self.state     = (0., 0., 0., 0.)
        self.k_ctrl    = 0

    def _c(self, a):
        cfg = self.cfg
        return cfg['C_MIN'] + (float(np.clip(a, -1., 1.)) + 1.) / 2. * (cfg['C_MAX'] - cfg['C_MIN'])

    def _obs(self):
        cfg = self.cfg
        x1, x1d, x2, x2d = self.state
        k  = min(self.k_ctrl * cfg['N_SUB'], self.n_phys - 1)
        zr = self._zr[k]
        return np.array([
            (x1-x2)/self._s_ref_ep,
            (x1d-x2d)/cfg['V_REF'],
            (x2-zr)/self._d_ref_ep,
            x1d/cfg['V_REF'],
            x2d/cfg['V_REF'],
            self._a_prev,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0., 0., 0., 0.)
        self.k_ctrl = 0
        self._a_prev = 0.
        cfg = self.cfg

        t_phys   = np.arange(self.n_phys) * cfg['DT']
        use_bump = (self.road_type == 'bump') or (
                    self.road_type == 'mixed' and float(self.np_random.random()) < 0.25)

        if options is not None and 'zr' in options:
            self._zr = options['zr']
        elif use_bump:
            h  = float(self.np_random.uniform(0.02, 0.08))
            w  = float(self.np_random.uniform(0.5, 2.0))
            v  = float(self.np_random.uniform(5., 20.))
            xs = float(self.np_random.uniform(1., 3.))
            self._zr = bump_profile(t_phys, v=v, h=h, width=w, x_start=xs)
        else:
            gq0_opts = [16e-6, 64e-6, 256e-6]
            Gq0 = gq0_opts[int(self.np_random.integers(0, 3))]
            v   = float(self.np_random.uniform(5., 20.))
            s   = int(self.np_random.integers(0, 99999))
            self._zr = iso_random_profile(t_phys, v=v, Gq0=Gq0, seed=s)

        passive_policy = lambda state, k: cfg['c0']
        yp_ep, x1dd_p, _ = rollout(self._zr, passive_policy, cfg)
        eps = 1e-6
        self._a_ref    = max(rms(x1dd_p),               eps)
        self._s_ref_ep = max(rms(yp_ep[0] - yp_ep[2]),  eps)
        self._d_ref_ep = max(rms(yp_ep[2] - self._zr),  eps)
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        a = float(np.clip(action[0], -1., 1.))
        c = self._c(a)
        k0 = self.k_ctrl * cfg['N_SUB']
        rew_sum = 0.
        for i in range(cfg['N_SUB']):
            k = k0 + i
            if k >= self.n_phys:
                break
            x1, x1d, x2, x2d = self.state
            zr_k = self._zr[k]
            self.state, x1dd, _ = step_sim(self.state, c, zr_k, cfg)
            susp = x1 - x2
            tyre = x2 - zr_k
            r = -((x1dd / self._a_ref)**2
                  + cfg['W2'] * (susp / self._s_ref_ep)**2
                  + cfg['W3'] * (tyre / self._d_ref_ep)**2)
            if abs(susp) > cfg['TRAVEL_LIMIT']:
                r -= cfg['W_LIM']
            rew_sum += r
        reward = rew_sum / cfg['N_SUB'] - cfg['W_DU'] * (a - self._a_prev)**2
        self._a_prev = a
        self.k_ctrl += 1
        return self._obs(), float(reward), self.k_ctrl >= self.n_ctrl, False, {}
