#!/bin/bash
# Nsight Compute wrapper script supporting rank targeting and kernel name filtering

# Target rank configuration: "0" (default), "all", or specific rank integer ("1", "2", etc.)
TARGET_RANK="${PROFILE_RANK:-0}"

# Build kernel filter argument (supports single kernel or regex "kernelA|kernelB")
KERNEL_FILTER=""
if [ -n "${NCU_KERNEL}" ]; then
    KERNEL_FILTER="-k regex:${NCU_KERNEL}"
fi

# Launch control options
SKIP_ARG=""
if [ -n "${NCU_SKIP}" ]; then
    SKIP_ARG="--launch-skip ${NCU_SKIP}"
fi
COUNT_ARG="--launch-count ${NCU_COUNT:-10}"

# Execute profiling for targeted rank(s), bypass for others
if [ "${TARGET_RANK}" = "all" ] || [ "${OMPI_COMM_WORLD_RANK}" -eq "${TARGET_RANK}" ]; then
    ncu \
        ${KERNEL_FILTER} \
        ${SKIP_ARG} \
        ${COUNT_ARG} \
        --target-processes all \
        -o "rhea_ncu_rank_%q{OMPI_COMM_WORLD_RANK}" \
        -f \
        "$@"
else
    "$@"
fi
