import numpy as np
from scipy.stats import lognorm
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from generate_corr_sequence import gen_corr_sequence

plt.rcParams.update({
    "figure.facecolor": (1.0, 1.0, 1.0, 1),
    "axes.facecolor": (1.0, 1.0, 1.0, 1),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1),
    "figure.dpi": 300,
    "figure.figsize": (4, 3),
    "font.size": 10,
})
# %% ACF
tau = np.arange(0, 12, 0.01)    # msec
L = 20                          # distance [m]
sigma = 0.2                     # Rytov variance
k = 2 * np.pi / 1.55e-6         # wavenumber
v = 1  # wind speed [m/s]
target_acf = (3.86 * 16 * sigma ** 2 * \
              np.imag(1j ** (1 / 6) * hyp1f1(-11 / 6, 1, -1j * k * (v * tau * 1e-3) ** 2 / (4 * L))) \
              - 7.52 * 16 * sigma ** 2 * (k * (v * tau * 1e-3) ** 2 / (4 * L)) ** (5 / 6)) / \
             (3.86 * 16 * sigma ** 2 * np.imag(1j ** (1 / 6) * hyp1f1(-11 / 6, 1, 0)))

plt.plot(tau, target_acf, label='Target ACF')
plt.grid()
plt.xlabel('Time (msec)')
plt.ylabel('ACF')
plt.xlim([0, 12])
plt.tight_layout()
plt.savefig('turbulence_acf.png')
plt.show()


# %%
dist_obj = lognorm(s=sigma, loc=0, scale=1)
signal = gen_corr_sequence(
    dist_obj=dist_obj,
    target_acf=target_acf,
    debug=True, plot_figures_name='turbulence.png')

