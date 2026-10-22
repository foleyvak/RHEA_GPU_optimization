# This script interpolates data between two meshes with the same physical domain. 
# The data files are named "input_data_file.h5" and "output_data_file.h5," and the interpolated data will be saved in a file named "interpolated_data_file.h5".

import sys
import os
import numpy as np
import h5py

### Set name of h5 files
input_name_data_file        = 'input_data_file.h5'          # Mesh to interpolate from
output_name_data_file       = 'output_data_file.h5'         # Mesh to interpolate to
interpolated_name_data_file = 'interpolated_data_file.h5'   # Output file
reset_attributes            = True  # Set to True to reset attributes, set to False otherwise

# Start timing
start_time = time.time()

print("Opening files ...")
with h5py.File(input_name_data_file, 'r') as input_file, \
     h5py.File(output_name_data_file, 'r') as output_file, \
     h5py.File(interpolated_name_data_file, 'w') as interpolated_file:

    # Read input and output mesh coordinates
    print("Reading mesh coordinates ...")
    input_x = input_file['x'][0, 0, :]
    input_y = input_file['y'][0, :, 0]
    input_z = input_file['z'][:, 0, 0]
    output_x = output_file['x'][0, 0, :]
    output_y = output_file['y'][0, :, 0]
    output_z = output_file['z'][:, 0, 0]

    # Pre-compute closest indices using np.searchsorted
    print("Precomputing closest indices ...")
    index_x_vector = np.searchsorted(input_x, output_x, side="left")
    index_y_vector = np.searchsorted(input_y, output_y, side="left")
    index_z_vector = np.searchsorted(input_z, output_z, side="left")

    # Clip indices to ensure they are within valid bounds
    index_x_vector = np.clip(index_x_vector, 0, len(input_x) - 1)
    index_y_vector = np.clip(index_y_vector, 0, len(input_y) - 1)
    index_z_vector = np.clip(index_z_vector, 0, len(input_z) - 1)

    print("Closest indices computed.")

    # Copy attributes
    print("Copying attributes ...")
    if reset_attributes:
        interpolated_file.attrs.create('Time', 0.0)
        interpolated_file.attrs.create('Iteration', 0)
        interpolated_file.attrs.create('AveragingTime', 0.0)
    else:
        for attr in ['Time', 'Iteration', 'AveragingTime']:
            interpolated_file.attrs.create(attr, input_file.attrs[attr])

    # Create datasets in the interpolated file
    print("Creating datasets in the interpolated file ...")
    for ds_name in output_file.keys():
        interpolated_file.create_dataset(ds_name, data=np.zeros_like(output_file[ds_name]))

    # Interpolation
    total_points = len(index_x_vector) * len(index_y_vector) * len(index_z_vector)
    print(f"Starting interpolation for {total_points} grid points ...")

    for dataset_idx, ds_name in enumerate(output_file.keys()):
        print(f"Interpolating dataset '{ds_name}' ...")
        
        # Load data into memory for faster access
        input_data = input_file[ds_name][:]
        interpolated_data = interpolated_file[ds_name]
        
        # Use advanced indexing to map all output points at once
        interpolated_data[:, :, :] = input_data[
            np.ix_(index_z_vector, index_y_vector, index_x_vector)
        ]
        
        # Update progress counter after completing one dataset
        print("Processed {} out of {} variables.".format(dataset_idx+1, len(output_file.keys())))
print("Interpolation completed.")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script finished. Total runtime: {elapsed_time:.2f} seconds.")