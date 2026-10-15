#!/bin/bash
#SBATCH --job-name=RHEA
#SBATCH --account=upc119
##SBATCH --qos=acc_debug
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
##SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=rhea_%j.out
#SBATCH --error=rhea_%j.err

module load nvidia-hpc-sdk/26.5 hdf5/1.14.1-2-nvidia-nvhpc yaml-cpp
export OMP_NUM_THREADS=1 # Only needed in BSC due to minimum number of CPU cores per requested GPU

# ======================================================================
# DEFAULT MODE: submit a non interactive job ->  sbatch run_rhea.sh 
#
# PRE-ALLOCATION MODE: 
#	1) run ./pre_allocation.sh; 
#	2) Once inside the allocation do ./run_rhea.sh
#
# PROFILING MODE:
#	1) Comment the "Normal execution" line
#	2) Uncomment the specific profiling line (Nsight systems or Compute)
#	3) ./run_rhea.sh
#
# ======================================================================

# ================== Normal execution =======================
# 4 MPI ranks = np_x * np_y * np_z in configuration_file.yaml (2 x 2 x 1)
# Each rank picks its own GPU inside the code (local_rank % num_devices).

#mpirun --bind-to none -np 4 ./RHEA.exe configuration_file.yaml

# =========== Nsight Systems profiling execution ==============
# PROFILE_RANK=all if profiling all ranks, or "0", "1", "2", "3" or any rank integer for specific rank profiling

PROFILE_RANK=0 mpirun --bind-to none -np 4 ./profile_nsys.sh ./RHEA.exe configuration_file.yaml

# =========== Nsight Compute profiling execution==============
# Example with all the options activated :
# PROFILE_RANK=0 NCU_KERNEL="calculateSourceTerms|timeAdvanceVelocityPointParticles" NCU_SKIP=20 NCU_COUNT=5 mpirun --bind-to none -np 4 ./profile_ncu.sh ./RHEA.exe configuration_file.yaml

#NCU_KERNEL="calculateViscousFluxes" mpirun --bind-to none -np 4 ./profile_ncu.sh ./RHEA.exe configuration_file.yaml


# ===========================================================

#NOTES FOR NSIGHT COMPUTE
# ENVIRONMENT VARIABLES REFERENCE:
#   PROFILE_RANK : Target rank(s) to profile.
#                  - Unset / "0" : Profile Rank 0 only (Default)
#                  - "all"       : Profile ALL MPI ranks
#                  - "1", "2"... : Profile a specific rank integer
#
#   NCU_KERNEL   : Filter by kernel name (supports single string or regex with '|').
#                  - Unset       : Profile ALL kernel launches
#                  - "name"      : Profile only kernels containing "name"
#                  - "A|B"       : Profile kernels containing "A" OR "B"
#
#   NCU_SKIP     : Number of matching kernel launches to SKIP before capturing data.
#                  - Unset / 0   : Start profiling immediately from launch #1
#                  - 20          : Ignore first 20 launches (skips setup/warm-up)
#
#   NCU_COUNT    : Number of matching kernel launches to PROFILE after skipping.
#                  - Unset / 10  : Capture 10 launches (Default)
#                  - 5           : Capture 5 launches

