import numpy as np


def bump_profile(t_arr, v=10.0, h=0.05, width=1.0, x_start=5.0):
    x = v * t_arr
    mask = (x >= x_start) & (x <= x_start + width)
    return np.where(mask, (h/2)*(1 - np.cos(2*np.pi*(x-x_start)/width)), 0.0)


def iso_random_profile(t_arr, v=10.0, Gq0=64e-6, seed=42):
    N = len(t_arr)
    dx = v * (t_arr[1] - t_arr[0])
    n_k = np.fft.rfftfreq(N, d=dx)
    n_k[0] = 1e-9
    n0 = 0.1
    Gq = Gq0 * (n_k / n0) ** -2
    Gq[0] = 0.0
    C_k = np.sqrt(2 * Gq / (N * dx))
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*np.pi, len(n_k))
    spectrum = (N / 2) * C_k * np.exp(1j * phases)
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=N)
