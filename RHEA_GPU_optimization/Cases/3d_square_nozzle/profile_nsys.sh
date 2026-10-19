#!/bin/bash
# Nsight Systems profiling wrapper script (rank to be set in the "run_rhea.sh" script)

TARGET_RANK="${PROFILE_RANK:-0}"

if [ "${TARGET_RANK}" = "all" ]; then
    # Profile all ranks with full MPI tracing
    nsys profile \
        --output="rhea_profile_rank_%q{OMPI_COMM_WORLD_RANK}" \
        --trace=cuda,openacc,nvtx,mpi \
        --force-overwrite=true \
        --stats=true \
        "$@"
elif [ "${OMPI_COMM_WORLD_RANK}" -eq "${TARGET_RANK}" ]; then
    # Profile single rank: omit 'mpi' trace to prevent PMPI deadlock at MPI_Finalize
    nsys profile \
        --output="rhea_profile_rank_${TARGET_RANK}" \
        --trace=cuda,openacc,nvtx \
        --force-overwrite=true \
        --stats=true \
        "$@"
else
    # Execute normally without profiling
    "$@"
fi
