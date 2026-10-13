#!/usr/bin/env bash
# Source only in the Orion NVIDIA GPU training job, before Python/Ray starts.
# Preserve Slurm's exact CUDA assignment; do not substitute physical GPU IDs.
printf 'GPU_VISIBILITY_BEFORE CUDA=%s ROCR=%s HIP=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" "${ROCR_VISIBLE_DEVICES-<unset>}" "${HIP_VISIBLE_DEVICES-<unset>}"
: "${CUDA_VISIBLE_DEVICES:?Expected Slurm CUDA_VISIBLE_DEVICES in this NVIDIA GPU job}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
printf 'GPU_VISIBILITY_READY CUDA=%s ROCR=unset HIP=unset\n' "$CUDA_VISIBLE_DEVICES"
