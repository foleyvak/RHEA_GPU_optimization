#!/usr/bin/python

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import rc,rcParams
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
from numpy import sin, cos, sqrt, ones, zeros, pi, arange
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve
from numpy.fft import fftn
plt.rc( 'text', usetex = True )
rc('font', family='sanserif')
plt.rc( 'font', size = 20 )
plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]

### Open data files
folder_path = './'
with os.scandir( folder_path ) as entries:
    file_list = sorted( [entry.name for entry in entries if entry.is_file() and entry.name.endswith( '.h5' )] )
time_list = np.array([]) 
for snapshot in (file_list) :
    dataset_name = "{}{}".format( folder_path, snapshot )
    with h5py.File( dataset_name, 'r' ) as data_file:
        time = data_file.attrs[ 'Time' ]
        time_list = np.append( time_list, np.array( time ) ) 
sorted_indices = np.argsort( time_list )
time_list = time_list[sorted_indices]
file_list = [file_list[i] for i in sorted_indices]
file_name = file_list[-1]           # Last snapshot
data_file = h5py.File( folder_path + file_name, 'r' )
u = data_file['u'][1:-1,1:-1,1:-1]
v = data_file['v'][1:-1,1:-1,1:-1]
w = data_file['w'][1:-1,1:-1,1:-1]
x_data = data_file['x'][:,:,:]
lx = ly = lz = ( x_data[0,0,-1] + x_data[0,0,-2] - x_data[0,0,1] - x_data[0,0,0] )/2.0


### Open reference solution file
reference_data = np.loadtxt('reference_solution.csv', skiprows = 21 )
k_eta_ref = reference_data[:, 0]
E_k_ref   = reference_data[:, 1]


### Reference parameters
P_0 = 101325.0                                                      # Reference bulk pressure
T_0 = 273.15                                                        # Reference bulk temperature
R_specific = 287.058                                                # Specific gas constant
rho_0 = P_0/(R_specific*T_0)                                        # Reference bulk density
mu_0 = 0.00001716                                                   # Dynamic viscosity
Ma_t = 0.049                                                        # Turbulent Mach number
Re_lambda = 38.0                                                    # Taylor Reynolds number
gamma_0 = 1.4                                                       # Heat capacity ratio
Re_L = ( Re_lambda**2.0 )/15.0                                            
u_0 = Ma_t*np.sqrt( gamma_0*R_specific*T_0/3.0 )
L_0 = ( mu_0/P_0 )*np.sqrt( 3.0*R_specific*T_0/gamma_0 )*( Re_L/Ma_t )
L_box = 0.00075
epsilon_0 = ( u_0**3.0 )/L_0
lambda_T = u_0/np.sqrt( rho_0*epsilon_0/( 15.0*mu_0 ) )
eta = ( ( ( mu_0/rho_0 )**3.0 )/epsilon_0 )**( 1.0/4.0 )


#################################################################
def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window, 'same')

def compute_tke_spectrum(u, v, w, lx, ly, lz, smooth):
    nx = len(u[:, 0, 0])
    ny = len(v[0, :, 0])
    nz = len(w[0, 0, :])

    nt = nx * ny * nz
    n = nx  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u) / nt
    vh = fftn(v) / nt
    wh = fftn(w) / nt

    tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)).real

    k0x = 2.0 * pi / lx
    k0y = 2.0 * pi / ly
    k0z = 2.0 * pi / lz

    knorm = (k0x + k0y + k0z) / 3.0
    #print('knorm = ', knorm)

    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2

    # dk = (knorm - kmax)/n
    # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

    wave_numbers = knorm * arange(0, n)

    tke_spectrum = zeros(len(wave_numbers))

    for kx in range(-nx//2, nx//2-1):
        for ky in range(-ny//2, ny//2-1):
            for kz in range(-nz//2, nz//2-1):
                rk = sqrt(kx**2 + ky**2 + kz**2)
                k = int(np.round(rk))
                tke_spectrum[k] += tkeh[kx, ky, kz]
    tke_spectrum = tke_spectrum / knorm

    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm * min(nx, ny, nz) / 2

    return knyquist, wave_numbers, tke_spectrum
#################################################################


### Calculate energy spectrum
knyquist, wavenumbers, tkespec = compute_tke_spectrum( u, v, w, lx, ly, lz, False )
    

### Plot normalized spectrum

# Clear plot
plt.clf()

# Plot data
plt.loglog( k_eta_ref, E_k_ref, linestyle = '-', linewidth = 2, marker = 's', markersize = 5, color = 'black', zorder = 0, label = r'$\textrm{Jimenez et al., }Re_\lambda = 36.4$' )
plt.loglog( wavenumbers*eta, tkespec/(epsilon_0**(2.0/3.0)*eta**(5.0/3.0)), linestyle = '-.', linewidth = 2, color = 'firebrick' , zorder = 1, label = r'$\textrm{RHEA}$' )

# Configure plot
plt.xlim( 5.0e-2, 5.0e0 )
plt.xticks( np.arange( 5.0e-2, 5.01, 1.0 ) )
plt.tick_params( axis = 'x', left = True, right = False, labelleft = True, labelright = False, direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( r'$k \eta$' )
plt.ylim( 1.0e-9, 1.0e3 )
plt.yticks( np.arange( 1.0e-9, 1.01e3, 1.0 ) )
plt.tick_params( axis = 'y', left = True, right = False, labelleft = True, labelright = False, direction = 'in' )
plt.yscale( 'log' )
plt.ylabel( r'$\epsilon^{-2/3}\eta^{-5/3}E(k)$' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='lower left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( 'normalized_energy_spectrum.eps', format = 'eps', bbox_inches = 'tight' )
