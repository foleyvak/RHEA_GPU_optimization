# This tool is designed to interpolate data from a coarse mesh to a fine mesh for two domains with the same real dimensions. 
# The input data files are named "coarse_data_file.h5" and "fine_data_file.h5," and the interpolated data will be saved in a file named "interpolated_data.h5".
# Script developed by Carlos Monteiro - Nov 2023

import sys
import os
import numpy as np
import h5py
import multiprocessing
from scipy.ndimage import gaussian_filter

# INSTRUCTIONS:
# To use the script, you need to have two domains with the same real dimensions, one with a coarse mesh and one with a finer mesh.
# Rename them as coarse_data_file.h5 and fine_data_file.h5

##### Parameters ######

# Specify the file names for coarse and fine datasets
coarse_dataset_name = 'coarse_data_file.h5'
fine_dataset_name = 'fine_data_file.h5'

# Output file for the interpolated data
interpolated_output_file_name = 'interpolated_data_file.h5'

# Output file for the filtered interpolated data
filtered_output_file_name = 'filtered_interpolated_data_file.h5'

# Define the number of cubes in each dimension 
num_cubes_x = 40
num_cubes_y = 40
num_cubes_z = 40

# Adjust the number of CPU cores to utilize
num_processes = 32  

##### Do not change below this part #######

# Initialize fields
format_dataset = np.float64
datasets_generated = []
processed_i_fine = set()

# Open the coarse and fine data files for reading
coarse_data_file = h5py.File(coarse_dataset_name, 'r')
fine_data_file = h5py.File(fine_dataset_name, 'r')

# Open spatial data from coarse mesh
coarse_x_data = coarse_data_file['x'][:, :, :]
coarse_y_data = coarse_data_file['y'][:, :, :]
coarse_z_data = coarse_data_file['z'][:, :, :]
num_coarse_points_x = coarse_x_data[0, 0, :].size
num_coarse_points_y = coarse_y_data[0, :, 0].size
num_coarse_points_z = coarse_z_data[:, 0, 0].size

# Open spatial data from fine mesh
fine_x_data = fine_data_file['x'][:, :, :]
fine_y_data = fine_data_file['y'][:, :, :]
fine_z_data = fine_data_file['z'][:, :, :]
num_fine_points_x = fine_x_data[0, 0, :].size
num_fine_points_y = fine_y_data[0, :, 0].size
num_fine_points_z = fine_z_data[:, 0, 0].size

# Create a new HDF5 file for the interpolated data
interpolated_output_file = h5py.File(interpolated_output_file_name, 'w')

# Copy time attributes from coarse to extruded file
attribute_time = coarse_data_file.attrs['Time']
interpolated_output_file.attrs.create('Time', attribute_time)
attribute_iteration = coarse_data_file.attrs['Iteration']
interpolated_output_file.attrs.create('Iteration', attribute_iteration)
attribute_averaging_time = coarse_data_file.attrs['AveragingTime']
interpolated_output_file.attrs.create('AveragingTime', attribute_averaging_time)

# Calculate the box size based on the domain size and the number of cubes
box_size_x = fine_x_data[num_fine_points_z - 1, num_fine_points_y - 1, num_fine_points_x - 1] / num_cubes_x
box_size_y = fine_y_data[num_fine_points_z - 1, num_fine_points_y - 1, num_fine_points_x - 1] / num_cubes_y
box_size_z = fine_z_data[num_fine_points_z - 1, num_fine_points_y - 1, num_fine_points_x - 1] / num_cubes_z

# Copy spatial data from the fine mesh to the output file
datasets = list(fine_data_file.keys())
for ds in datasets:
    fine_dataset = fine_data_file[ds][:, :, :]
    interpolated_output_file.create_dataset(ds, data=fine_dataset)
print('Initialized interpolated datasets with preliminary fine mesh data')

# Create a spatial index for the coarse mesh with the new box sizes
coarse_index = {}
for i in range(num_coarse_points_x):
    for j in range(num_coarse_points_y):
        for k in range(num_coarse_points_z):
            box_x = int(coarse_x_data[k, j, i] / box_size_x)
            box_y = int(coarse_y_data[k, j, i] / box_size_y)
            box_z = int(coarse_z_data[k, j, i] / box_size_z)
            if (box_x, box_y, box_z) not in coarse_index:
                coarse_index[(box_x, box_y, box_z)] = []
            coarse_index[(box_x, box_y, box_z)].append((i, j, k))
print('Created coarse boxes indexes')


def process_fine_point(i_fine, j_fine, k_fine):
    
    if i_fine not in processed_i_fine:
        print("New i_fine:", i_fine)
        processed_i_fine.add(i_fine)

    local_x_fine = fine_x_data[k_fine, j_fine, i_fine]
    local_y_fine = fine_y_data[k_fine, j_fine, i_fine]
    local_z_fine = fine_z_data[k_fine, j_fine, k_fine]
    min_distance = 100000  # Reset the minimum distance for each fine mesh point
    
    # Calculate the box coordinates for the fine mesh point
    box_x_fine = int(local_x_fine / box_size_x)
    box_y_fine = int(local_y_fine / box_size_y)
    box_z_fine = int(local_z_fine / box_size_z)

    # Search only within the same and neighboring boxes
    for box_x in range(box_x_fine - 1, box_x_fine + 1):
        for box_y in range(box_y_fine - 1, box_y_fine + 1):
            for box_z in range(box_z_fine - 1, box_z_fine + 1):
                if (box_x, box_y, box_z) in coarse_index:
                    for i_coarse, j_coarse, k_coarse in coarse_index[(box_x, box_y, box_z)]:
                        local_x_coarse = coarse_x_data[k_coarse, j_coarse, i_coarse]
                        local_y_coarse = coarse_y_data[k_coarse, j_coarse, i_coarse]
                        local_z_coarse = coarse_z_data[k_coarse, j_coarse, i_coarse]

                        # Calculate distance
                        distance = np.sqrt((local_x_fine - local_x_coarse) ** 2 +
                                           (local_y_fine - local_y_coarse) ** 2 +
                                           (local_z_fine - local_z_coarse) ** 2)

                        # Find closest point in the coarse mesh
                        if distance < min_distance:
                            i_closest = i_coarse
                            j_closest = j_coarse
                            k_closest = k_coarse
                            min_distance = distance
    
    # Copy all data in that point to the fine mesh point
    datasets = list(fine_data_file.keys())
    for ds in datasets:
        if (ds not in ['x', 'y', 'z']):
            interpolated_output_file[ds][k_fine, j_fine, i_fine] = coarse_data_file[ds][k_closest, j_closest, i_closest]      
            #print('updated dataset of (i,j,k)=',i_fine,j_fine,k_fine,'    ds=',ds,'   output=',interpolated_output_file[ds][k_fine, j_fine, i_fine])

# Process fine mesh points in parallel
pool = multiprocessing.Pool(processes=num_processes)
fine_point_args = [(i, j, k) for i in range(num_fine_points_x) for j in range(num_fine_points_y) for k in range(num_fine_points_z)]
pool.starmap(process_fine_point, fine_point_args)
pool.close()
pool.join()

# Close the original & output data files
coarse_data_file.close()
fine_data_file.close()
interpolated_output_file.close()

##### Apply Filter #####

def apply_filter(input_file_name, output_file_name, sigma=1.0):
    # Open the interpolated data file
    interpolated_data_file = h5py.File(input_file_name, 'r+')

    # Apply a Gaussian filter to each dataset in the interpolated file
    datasets = list(interpolated_data_file.keys())
    for ds in datasets:
        if ds not in ['x', 'y', 'z']:
            data = interpolated_data_file[ds][:, :, :]

            # Apply Gaussian filter with the specified sigma
            filtered_data = gaussian_filter(data, sigma=sigma)

            # Update the dataset with the filtered data
            interpolated_data_file[ds][:, :, :] = filtered_data

    # Close the interpolated data file
    interpolated_data_file.close()

    # Save the filtered interpolated data to a new file
    filtered_interpolated_data_file = h5py.File(output_file_name, 'w')
    
    # Copy datasets from the input file to the output file
    with h5py.File(input_file_name, 'r') as input_file:
        for key in input_file.keys():
            input_file.copy(key, filtered_interpolated_data_file)

    # Close the filtered interpolated data file
    filtered_interpolated_data_file.close()

if __name__ == '__main__':
    # Specify the input and output file names for the filtering step
    filter_input_file_name = 'interpolated_data.h5'
    filter_output_file_name = 'filtered_interpolated_data.h5'

    # Specify the desired sigma for the Gaussian filter (adjust as needed)
    filter_sigma = 1.0

    # Apply the filter and save the result
    apply_filter(filter_input_file_name, filter_output_file_name, sigma=filter_sigma)
