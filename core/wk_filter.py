import numpy as np
from scipy.signal import sosfilt, zpk2sos, sosfreqz, bilinear_zpk


def _stage(z_ct, p_ct, k_ct, fs):
    zd, pd, kd = bilinear_zpk(z_ct, p_ct, k_ct, fs)
    return zpk2sos(zd, pd, kd)


def make_wk_sos(fs=1000.):
    stages = []
    f1, Q1 = 0.4, 1./np.sqrt(2); w1 = 2*np.pi*f1
    stages.append(_stage([0., 0.], np.roots([1, w1/Q1, w1**2]), 1., fs))
    f2, Q2 = 100., 1./np.sqrt(2); w2 = 2*np.pi*f2
    stages.append(_stage([], np.roots([1, w2/Q2, w2**2]), w2**2, fs))
    f3 = f4 = 12.5; Q4 = 0.63; w3 = 2*np.pi*f3; w4 = 2*np.pi*f4
    stages.append(_stage([-w3], np.roots([1, w4/Q4, w4**2]), w4**2/w3, fs))
    f5, Q5, f6, Q6 = 2.37, 0.91, 3.35, 0.91
    w5 = 2*np.pi*f5; w6 = 2*np.pi*f6
    stages.append(_stage(np.roots([1, w5/Q5, w5**2]),
                          np.roots([1, w6/Q6, w6**2]), w6**2/w5**2, fs))
    return np.vstack(stages)


def wk_rms(accel, sos=None, fs=1000.):
    if sos is None:
        sos = make_wk_sos(fs)
    from core.dynamics import rms
    return rms(sosfilt(sos, accel))
