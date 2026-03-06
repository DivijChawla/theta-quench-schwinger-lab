#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/scrubbed/${USER}/theta_quench_magic_lab}"
PARTITION_GPU="${PARTITION_GPU:-ckpt-all}"
PARTITION_CPU="${PARTITION_CPU:-compute}"
ACCOUNT_CPU="${ACCOUNT_CPU:-stf}"
ACCOUNT_GPU="${ACCOUNT_GPU:-stf-ckpt}"
QOS_GPU="${QOS_GPU:-ckpt}"

cd "$PROJECT_ROOT"
mkdir -p slurm_logs

setup_job=$(sbatch --parsable --account="$ACCOUNT_CPU" --partition="$PARTITION_CPU" cluster/hyak/setup_env.slurm)
stage2_job=$(sbatch --parsable --dependency=afterok:${setup_job} --account="$ACCOUNT_GPU" --partition="$PARTITION_GPU" --qos="$QOS_GPU" cluster/hyak/stage2_ckpt.slurm)
stage4_job=$(sbatch --parsable --dependency=afterok:${setup_job} --account="$ACCOUNT_GPU" --partition="$PARTITION_GPU" --qos="$QOS_GPU" cluster/hyak/stage4_ckpt.slurm)
stage5_job=$(sbatch --parsable --dependency=afterok:${setup_job} --account="$ACCOUNT_GPU" --partition="$PARTITION_GPU" --qos="$QOS_GPU" cluster/hyak/stage5_ckpt.slurm)

echo "Submitted:"
echo "  setup_env  ${setup_job}"
echo "  stage2     ${stage2_job} (afterok:${setup_job})"
echo "  stage4     ${stage4_job} (afterok:${setup_job})"
echo "  stage5     ${stage5_job} (afterok:${setup_job})"
echo
echo "Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f ${PROJECT_ROOT}/slurm_logs/tqm_setup_env_${setup_job}.out"
