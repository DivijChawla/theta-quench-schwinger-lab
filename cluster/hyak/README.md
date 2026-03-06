# Hyak Runbook

This repo is now Hyak-ready for the heavy NNQS and cross-family studies.

Use Hyak for:

- larger seed sweeps,
- denser parameter grids,
- stage-2 / stage-3 / stage-5 cross-family runs,
- GPU-backed NNQS training with `--device auto` or `--device cuda`.

Do not use Hyak login nodes for heavy jobs. Follow the Hyak docs workflow:

1. `ssh dc245@klone.hyak.uw.edu`
2. `hyakalloc`
3. keep working data under `/gscratch/...`, not in home
4. submit with `sbatch`
5. monitor with `squeue -u dc245`

## Recommended remote layout

```bash
mkdir -p /gscratch/scrubbed/$USER/theta_quench_magic_lab
```

## Local-to-Hyak sync

From the local repo root:

```bash
bash cluster/hyak/rsync_to_klone.sh
```

This copies code/configs/docs to the remote project directory without deleting existing remote outputs.

## Bootstrap the remote environment

After logging into Klone:

```bash
cd /gscratch/scrubbed/$USER/theta_quench_magic_lab
bash cluster/hyak/bootstrap_env.sh
```

This script prefers a Python 3.11 conda env at `~/.conda-envs/theta-quench-magic-lab`, which avoids the too-old `/usr/bin/python3` on Klone login nodes and keeps the environment off `/gscratch/scrubbed`, where a direct env create left a partial install.

## Submit jobs

Current live `hyakalloc` output for this account shows:

- account: `stf`
- GPU partitions: `gpu-l40s`, `gpu-l40`, `gpu-2080ti`
- best default right now: `gpu-l40s`

The Slurm scripts therefore default to `stf` on `gpu-l40s` with one GPU. Override at submit time if queue pressure changes:

```bash
sbatch --account=stf --partition=gpu-l40s cluster/hyak/stage2_ckpt.slurm
sbatch --account=stf --partition=gpu-l40s cluster/hyak/stage4_ckpt.slurm
sbatch --account=stf --partition=gpu-l40s cluster/hyak/stage5_ckpt.slurm
```

## Monitoring

```bash
squeue -u dc245
tail -f /gscratch/scrubbed/$USER/theta_quench_magic_lab/slurm_logs/<jobname>_<jobid>.out
```

## What Hyak actually improves here

Hyak helps most on the NNQS-heavy and multi-seed parts of the study. Exact dynamics and exact magic remain exponential, so stronger claims come from pairing Hyak with:

- more seeds and bootstrap power,
- more architectures / control runs,
- denser regime scans,
- approximate large-`N` magic estimators,
- tensor-network or symmetry-reduced dynamics for larger systems.

## GPU selection note

The current code uses a single Python process and only one NNQS device per run. That means:

- request `1` GPU per job,
- prefer several independent jobs over `3` GPUs in one job,
- use `gpu-l40s` by default unless queue pressure favors `gpu-l40` or `gpu-2080ti`.
