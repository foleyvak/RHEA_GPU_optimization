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
#np.set_printoptions(threshold=sys.maxsize)
plt.rc( 'text', usetex = True )
rc('font', family='sanserif')
plt.rc( 'font', size = 20 )
plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]


### Open data file
data_file = h5py.File( '3d_ibm_circular_cylinder_1000000.h5', 'r' )
#list( data_file.keys() )
x_data       = data_file['x'][:,:,:]
y_data       = data_file['y'][:,:,:]
#z_data       = data_file['z'][:,:,:]
avg_u_data   = data_file['avg_u'][:,:,:]
#avg_v_data   = data_file['avg_v'][:,:,:]
#avg_w_data   = data_file['avg_w'][:,:,:]
#rmsf_u_data  = data_file['rmsf_u'][:,:,:]
#rmsf_v_data  = data_file['rmsf_v'][:,:,:]
#rmsf_w_data  = data_file['rmsf_w'][:,:,:]
num_points_x = avg_u_data[0,0,:].size
num_points_y = avg_u_data[0,:,0].size
num_points_z = avg_u_data[:,0,0].size


### Open reference solution files
avg_x_centerline_parnaudeau_ref, avg_u_centerline_parnaudeau_ref = np.loadtxt( 'parnaudeau_normalized_u_vs_x_centerline.csv', delimiter=',', unpack = 'True' )
avg_y_profiles_parnaudeau_ref, avg_u_profile_106_parnaudeau_ref, avg_u_profile_154_parnaudeau_ref, avg_u_profile_202_parnaudeau_ref = np.loadtxt( 'parnaudeau_normalized_u_vs_y_streamwise_profiles.csv', delimiter=',', unpack = 'True' )
avg_x_centerline_lourenco_ref, avg_u_centerline_lourenco_ref = np.loadtxt( 'lourenco_normalized_u_vs_x_centerline.csv', delimiter=',', unpack = 'True' )
avg_y_profiles_lourenco_ref, avg_u_profile_106_lourenco_ref, avg_u_profile_154_lourenco_ref, avg_u_profile_202_lourenco_ref = np.loadtxt( 'lourenco_normalized_u_vs_y_streamwise_profiles.csv', delimiter=',', unpack = 'True' )


### Reference parameters
rho_ref = 1.0				# Reference density [kg/m3]
u_infty = 1.0				# Free-stream velocity [m/s]
D_cyl   = 1.0				# Diameter of cylinder [m]
Re_D    = 3900.0			# Reynolds number [-]
mu_ref  = rho_ref*u_infty*D_cyl/Re_D	# Dynamic viscosity [Pa s]


### Allocate averaged variables
raw_avg_x_centerline = np.zeros( num_points_x - 2 )
raw_avg_u_centerline = np.zeros( num_points_x - 2 )
avg_y_profile_106    = np.zeros( num_points_y - 2 )
avg_y_profile_154    = np.zeros( num_points_y - 2 )
avg_y_profile_202    = np.zeros( num_points_y - 2 )
avg_u_profile_106    = np.zeros( num_points_y - 2 )
avg_u_profile_154    = np.zeros( num_points_y - 2 )
avg_u_profile_202    = np.zeros( num_points_y - 2 )


### Average variables in space
num_points_avg = num_points_z - 2
for i in range( 1, num_points_x - 1 ):
    for j in range( 1, num_points_y - 1 ):
        for k in range( 1, num_points_z - 1 ):
            x_position = x_data[k,j,i]/D_cyl
            y_position = y_data[k,j,i]/D_cyl
            #z_position = z_data[k,j,i]/D_cyl
            Delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )/D_cyl
            Delta_y = 0.5*( y_data[k,j+1,i] - y_data[k,j-1,i] )/D_cyl
            #Delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )/D_cyl
            if( ( y_position > (-1.0)*0.5*Delta_y ) and ( y_position < 0.5*Delta_y ) ):
                raw_avg_x_centerline[i-1] += ( 1.0/num_points_avg )*x_data[k,j,i]*( 1.0/D_cyl )
                raw_avg_u_centerline[i-1] += ( 1.0/num_points_avg )*avg_u_data[k,j,i]*( 1.0/u_infty )
            if( ( x_position > ( 1.06 - 0.5*Delta_x ) ) and ( x_position < ( 1.06 + 0.5*Delta_x ) ) ):
                avg_y_profile_106[j-1] += ( 1.0/num_points_avg )*y_data[k,j,i]*( 1.0/D_cyl )
                avg_u_profile_106[j-1] += ( 1.0/num_points_avg )*avg_u_data[k,j,i]*( 1.0/u_infty )
            if( ( x_position > ( 1.54 - 0.5*Delta_x ) ) and ( x_position < ( 1.54 + 0.5*Delta_x ) ) ):
                avg_y_profile_154[j-1] += ( 1.0/num_points_avg )*y_data[k,j,i]*( 1.0/D_cyl )
                avg_u_profile_154[j-1] += ( 1.0/num_points_avg )*avg_u_data[k,j,i]*( 1.0/u_infty )
            if( ( x_position > ( 2.02 - 0.5*Delta_x ) ) and ( x_position < ( 2.02 + 0.5*Delta_x ) ) ):
                avg_y_profile_202[j-1] += ( 1.0/num_points_avg )*y_data[k,j,i]*( 1.0/D_cyl )
                avg_u_profile_202[j-1] += ( 1.0/num_points_avg )*avg_u_data[k,j,i]*( 1.0/u_infty )
#print( raw_avg_x_centerline )
#print( raw_avg_u_centerline )
#print( avg_y_profile_106 )
#print( avg_y_profile_154 )
#print( avg_y_profile_202 )
#print( avg_u_profile_106 )
#print( avg_u_profile_154 )
#print( avg_u_profile_202 )


### Remove points inside cylinder
avg_x_centerline = []
avg_u_centerline = []
for i in range( 1, num_points_x - 1 ):
    if( raw_avg_x_centerline[i-1] > 0.5 ):
        avg_x_centerline.append( raw_avg_x_centerline[i-1] )
        avg_u_centerline.append( raw_avg_u_centerline[i-1] )


### Plot u/u_infty vs. x/D centerline

# Clear plot
plt.clf()

# Read & Plot data
plt.scatter( avg_x_centerline_parnaudeau_ref[::4], avg_u_centerline_parnaudeau_ref[::4], marker = 'v', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Parnaudeau et al. 2008}$' )
plt.scatter( avg_x_centerline_lourenco_ref[::1], avg_u_centerline_lourenco_ref[::1], marker = '^', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Lourenco \& Shih 1993}$' )
plt.plot( avg_x_centerline, avg_u_centerline, linestyle = '-.', linewidth = 2, color = 'firebrick', zorder = 1, label = r'$\textrm{RHEA}$' )

# Configure plot
plt.text( 3.5, -0.375, r'$y/D = 0.0$' )
plt.xlim( 0.0, 5.0 )
plt.xticks( np.arange( 0.0, 5.01, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
#plt.xscale( 'log' )
plt.xlabel( r'$x/D$' )
plt.ylim( -0.5, 1.25 )
plt.yticks( np.arange( -0.5, 1.26, 0.25 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( r'$\overline{u}/u_{\infty}$' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( 'normalized_u_vs_x_centerline.eps', format = 'eps', bbox_inches = 'tight' )


### Plot u/u_infty vs. y/D streamwise profile x/D = 1.06

# Clear plot
plt.clf()

# Read & Plot data
plt.scatter( avg_y_profiles_parnaudeau_ref[::2], avg_u_profile_106_parnaudeau_ref[::2], marker = 'v', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Parnaudeau et al. 2008}$' )
plt.scatter( avg_y_profiles_lourenco_ref[::1], avg_u_profile_106_lourenco_ref[::1], marker = '^', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Lourenco \& Shih 1993}$' )
plt.plot( avg_y_profile_106, avg_u_profile_106, linestyle = '-.', linewidth = 2, color = 'firebrick', zorder = 1, label = r'$\textrm{RHEA}$' )

# Configure plot
plt.text( 0.75, -0.25, r'$x/D = 1.06$' )
plt.xlim( -2.0, 2.0 )
plt.xticks( np.arange( -2.0, 2.01, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
#plt.xscale( 'log' )
plt.xlabel( r'$y/D$' )
plt.ylim( -0.5, 2.5 )
plt.yticks( np.arange( -0.5, 2.51, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( r'$\overline{u}/u_{\infty}$' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( 'normalized_u_vs_y_streamwise_profile_106.eps', format = 'eps', bbox_inches = 'tight' )


### Plot u/u_infty vs. y/D streamwise profile x/D = 1.54

# Clear plot
plt.clf()

# Read & Plot data
plt.scatter( avg_y_profiles_parnaudeau_ref[::2], avg_u_profile_154_parnaudeau_ref[::2], marker = 'v', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Parnaudeau et al. 2008}$' )
plt.scatter( avg_y_profiles_lourenco_ref[::1], avg_u_profile_154_lourenco_ref[::1], marker = '^', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Lourenco \& Shih 1993}$' )
plt.plot( avg_y_profile_154, avg_u_profile_154, linestyle = '-.', linewidth = 2, color = 'firebrick', zorder = 1, label = r'$\textrm{RHEA}$' )

# Configure plot
plt.text( 0.75, -0.25, r'$x/D = 1.54$' )
plt.xlim( -2.0, 2.0 )
plt.xticks( np.arange( -2.0, 2.01, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
#plt.xscale( 'log' )
plt.xlabel( r'$y/D$' )
plt.ylim( -0.5, 2.5 )
plt.yticks( np.arange( -0.5, 2.51, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( r'$\overline{u}/u_{\infty}$' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( 'normalized_u_vs_y_streamwise_profile_154.eps', format = 'eps', bbox_inches = 'tight' )


### Plot u/u_infty vs. y/D streamwise profile x/D = 2.02

# Clear plot
plt.clf()

# Read & Plot data
plt.scatter( avg_y_profiles_parnaudeau_ref[::2], avg_u_profile_202_parnaudeau_ref[::2], marker = 'v', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Parnaudeau et al. 2008}$' )
plt.scatter( avg_y_profiles_lourenco_ref[::1], avg_u_profile_202_lourenco_ref[::1], marker = '^', s = 50, color = 'black', zorder = 0, label = r'$\textrm{Lourenco \& Shih 1993}$' )
plt.plot( avg_y_profile_202, avg_u_profile_202, linestyle = '-.', linewidth = 2, color = 'firebrick', zorder = 1, label = r'$\textrm{RHEA}$' )

# Configure plot
plt.text( 0.75, -0.25, r'$x/D = 2.02$' )
plt.xlim( -2.0, 2.0 )
plt.xticks( np.arange( -2.0, 2.01, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
#plt.xscale( 'log' )
plt.xlabel( r'$y/D$' )
plt.ylim( -0.5, 2.5 )
plt.yticks( np.arange( -0.5, 2.51, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( r'$\overline{u}/u_{\infty}$' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( 'normalized_u_vs_y_streamwise_profile_202.eps', format = 'eps', bbox_inches = 'tight' )
