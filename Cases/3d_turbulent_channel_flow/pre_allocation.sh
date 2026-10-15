#!/bin/bash

# Submit a pre-allocation job by doing ./pre_allocation.sh

# Script to read settings from (defaults to run_rhea.sh if $1 is omitted)
TARGET_SCRIPT="${1:-run_rhea.sh}"

# Helper function to extract #SBATCH option values
get_sbatch_val() {
  grep -E "^#SBATCH --$1=" "$2" 2>/dev/null | cut -d'=' -f2
}

# Extract GPU count from the --gres=gpu:X directive
GPUS=$(grep -E "^#SBATCH --gres=gpu:" "$TARGET_SCRIPT" 2>/dev/null | sed -E 's/.*gpu:([0-9]+).*/\1/')

# Parse parameters dynamically from the target script
ALLOC_TIME="00:30:00"     # Duration of the allocation (HH:MM:SS), must respect the QOS limit

ACCOUNT=$(get_sbatch_val "account" "$TARGET_SCRIPT")
QOS=$(get_sbatch_val "qos" "$TARGET_SCRIPT")
NNODES=$(get_sbatch_val "nodes" "$TARGET_SCRIPT")
NTASKS=$(get_sbatch_val "ntasks" "$TARGET_SCRIPT")
CPUS_PER_TASK=$(get_sbatch_val "cpus-per-task" "$TARGET_SCRIPT")

#ACCOUNT="upc119"
#QOS="acc_resa"
#NNODES=1
#NTASKS=4
#CPUS_PER_TASK=20
#GPU=4

salloc --account=$ACCOUNT \
       --qos=$QOS \
       --nodes=$NNODES \
       --ntasks=$NTASKS \
       --ntasks-per-node=$NTASKS \
       --cpus-per-task=$CPUS_PER_TASK \
       --gres=gpu:$GPUS \
       --time=$ALLOC_TIME \
       --exclusive \
       --job-name=PreAlloc \
       $( [ -n "$1" ] && echo "bash $1" )
