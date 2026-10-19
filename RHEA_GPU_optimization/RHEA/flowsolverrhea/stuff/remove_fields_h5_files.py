import os
import h5py
import re  # For sorting filenames
import time  # Import time module


# Define datasets to keep in reduced snapshots
datasets_to_keep = ['x', 'y', 'z', 'u', 'v', 'w'] 

def process_snapshot(file_path, output_folder):
    """Processes an HDF5 snapshot, keeping only specified datasets."""
    file_name = os.path.basename(file_path)
    new_file_name = os.path.join(output_folder, file_name)

    print(f"Processing {file_name}...")

    with h5py.File(file_path, 'r') as data_file, h5py.File(new_file_name, 'w') as new_file:
        for dataset_name in datasets_to_keep:
            if dataset_name in data_file:
                new_file.create_dataset(dataset_name, data=data_file[dataset_name][:])
                print(f"  → Copied dataset: {dataset_name}")

        for attr_name, attr_value in data_file.attrs.items():
            new_file.attrs[attr_name] = attr_value

    print(f"Finished processing {file_name}, saved as {new_file_name}\n")
    return new_file_name

def generate_xdmf(hdf5_file_path):
    """Generates an XDMF file for visualization."""
    """Generates a correctly formatted XDMF file with separate Geometry and Attributes sections."""
    file_name = os.path.basename(hdf5_file_path)  # Extract only the filename
    xdmf_file_path = hdf5_file_path.replace('.h5', '.xdmf')

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Get dataset dimensions (assuming all datasets have the same shape)
        first_dataset = list(hdf5_file.keys())[0]
        dimensions = ' '.join(map(str, hdf5_file[first_dataset].shape))

        # XDMF header
        xdmf_content = f"""<?xml version='1.0' ?>
<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>
<Xdmf Version='2.0'>
  <Domain> 
    <Grid Name='{file_name.replace(".h5", "")}' GridType='Uniform'>
      <Topology TopologyType='3DSMesh' Dimensions='{dimensions}'/>
      <Geometry GeometryType='X_Y_Z'>
"""

        # Add geometry fields (x, y, z)
        for dim in ['x', 'y', 'z']:
            if dim in hdf5_file:
                xdmf_content += f"""        <DataItem Name='{dim}' Dimensions='{dimensions}' NumberType='Float' Precision='8' Format='HDF'>
            {file_name}:/{dim}
        </DataItem>\n"""

        xdmf_content += """      </Geometry>\n"""

        # Add all other fields as attributes
        for name in hdf5_file:
            if name not in ['x', 'y', 'z']:  # Skip geometry fields
                xdmf_content += f"""      <Attribute Name='{name}' AttributeType='Scalar' Center='Node'>
        <DataItem Dimensions='{dimensions}' NumberType='Float' Precision='8' Format='HDF'>
          {file_name}:/{name}
        </DataItem>
      </Attribute>\n"""

        # Close XDMF structure
        xdmf_content += """    </Grid>
  </Domain>
</Xdmf>"""

        # Write to file
        with open(xdmf_file_path, 'w') as xdmf_file:
            xdmf_file.write(xdmf_content)

    print(f"XDMF generated: {xdmf_file_path}\n")

def main(input_folder):
    """Processes all snapshots and extracts average fields from the latest one."""
    start_time = time.time()  # Start timer
    parent_folder = os.path.dirname(input_folder)
    output_folder = os.path.join(parent_folder, os.path.basename(input_folder) + "_reduced")

    print(f"Creating output folder: {output_folder}\n")
    os.makedirs(output_folder, exist_ok=True)

    processed_files = []
    for file in sorted(os.listdir(input_folder)):
        if file.endswith('.h5'):
            file_path = os.path.join(input_folder, file)
            new_file_path = process_snapshot(file_path, output_folder)
            generate_xdmf(new_file_path)
            processed_files.append(file_path)

    print("\nAll snapshots processed successfully!")
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute elapsed time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    input_folder = os.getcwd()  # Current directory where script is run
    main(input_folder)


