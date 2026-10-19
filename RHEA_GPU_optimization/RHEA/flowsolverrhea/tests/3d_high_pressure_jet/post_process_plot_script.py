import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Unified LaTeX style ===
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

D_jet = 0.0022

# === Load RHEA simulation data ===
with h5py.File('3d_hp_jet_4000000.h5', 'r') as data_file:
    x_data = data_file['x'][:] / D_jet
    rho_data = data_file['avg_rho'][:]
    nz, ny, nx = rho_data.shape

# === Extract centerline normalized density profile ===
x_line = x_data[nz//2, ny//2, :]
rho_ref = rho_data[nz-3, ny-3, nx//2]
rho_norm = (rho_data[nz//2, ny//2, :] - rho_ref) / (rho_data[nz//2, ny//2, 0] - rho_ref)

# Limit domain to x/D < 30
mask = x_line < 30
x_line = x_line[mask]
rho_norm = rho_norm[mask]

# Smooth with Savitzky-Golay filter
window_length = 11
polyorder = 2
rho_smooth = savgol_filter(rho_norm, window_length, polyorder) if len(rho_norm) >= window_length else rho_norm

# === Load Mayer experimental data (scatter) ===
x_mayer_exp, rho_mayer_exp = np.loadtxt( 'mayer_case_4_experimental_data_density.csv', delimiter=',', unpack = 'True' )

# === Plot ===
fig, ax = plt.subplots(figsize=(4.5, 3))

# RHEA (line)
ax.plot(x_line, rho_smooth, color='firebrick', linewidth=1.5, label='RHEA')

# Mayer experiment (scatter)
ax.scatter(x_mayer_exp, rho_mayer_exp, color='black', s=10, marker='x', label='Mayer (Case 4) experiment')

# Axis formatting
ax.set_xlabel(r"$x/D_{\mathrm{jet}}$")
ax.set_ylabel(r"$\rho^*$")
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.2)
ax.tick_params(direction='in', length=4, width=0.8, pad=4, top=True, right=True)
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

ax.legend(frameon=False, loc='best', handlelength=2.0, borderaxespad=0.5)
fig.tight_layout()
fig.savefig("normalized_rho_vs_axial_direction.eps", format='eps')
