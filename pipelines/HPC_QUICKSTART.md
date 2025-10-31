# ELIE on HPC

For running ELIE pipelines on High-Performance Computing clusters using Apptainer/Singularity.

> **macOS Users:** Apptainer requires a Linux kernel and does not work natively on macOS. You must build and run Apptainer containers on a Linux system or HPC cluster. For local development on macOS, use Docker instead.

## Why Apptainer?

- **No root required** - Runs without privileges
- **Secure** - Better security model than Docker for shared systems
- **HPC-native** - Works with SLURM, PBS, LSF schedulers
- **Compatible** - Works with Singularity installations

## Quick Setup

### 1. Build Container

```bash
cd pipelines
apptainer build elie.sif elie.def
```

This creates a single `elie.sif` file (~2-3 GB) containing the entire pipeline.

### 2. Transfer to HPC

```bash
# From your local machine
scp pipelines/elie.sif username@hpc.cluster.edu:/path/on/hpc/
```

### 3. Run Pipeline

```bash
# On HPC cluster
apptainer run --bind /scratch/data:/app/data elie.sif mli
```

## Usage

### Interactive Mode

```bash
# Start interactive session
salloc --mem=16G --cpus-per-task=8 --time=2:00:00

# Run pipeline
apptainer run --bind $(pwd)/data:/app/data elie.sif mli
```

### Batch Mode (SLURM)

**Option 1: Use provided template**

```bash
# Edit configuration
cp tools/hpc/run_elie_slurm.sh my_job.sh
nano my_job.sh  # Edit CONTAINER_PATH, INPUT_DATA_DIR, OUTPUT_DATA_DIR

# Submit
sbatch my_job.sh
```

**Option 2: Quick one-liner**

```bash
srun --mem=16G --cpus-per-task=8 \
  apptainer run --bind /scratch/data:/app/data elie.sif mli
```

### Job Arrays (Process Multiple Datasets)

```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --mem=16G
#SBATCH --time=02:00:00

INPUT=/scratch/batch_${SLURM_ARRAY_TASK_ID}/input
OUTPUT=/scratch/batch_${SLURM_ARRAY_TASK_ID}/output

INPUT_DIR=/app/data/input OUTPUT_DIR=/app/data/output \
  apptainer run \
  --bind $INPUT:/app/data/input \
  --bind $OUTPUT:/app/data/output \
  elie.sif mli
```

## Pipeline Types

### MLI (Multi-Label Images)
Full specimen photos with multiple labels - system detects and crops automatically.

```bash
apptainer run --bind /scratch/data:/app/data elie.sif mli
```

### SLI (Single-Label Images)
Pre-cropped individual label images.

```bash
apptainer run --bind /scratch/data:/app/data elie.sif sli
```

## Individual Commands

### Run Specific Stage

```bash
# Detection only
apptainer run --app detection --bind ./data:/app/data elie.sif \
  -j /app/data/MLI/input -o /app/data/MLI/output

# Rotation only
apptainer run --app rotation --bind ./data:/app/data elie.sif \
  -i /app/data/input -o /app/data/output

# OCR only
apptainer run --app tesseract --bind ./data:/app/data elie.sif \
  -d /app/data/input -o /app/data/output
```

### Use Exec for Any Script

```bash
apptainer exec --bind ./data:/app/data elie.sif \
  python /app/scripts/processing/classifiers.py -m 1 -j /app/data/input -o /app/data/output
```

## Data Management

### Directory Structure on HPC

```
/scratch/$USER/elie/
├── input/              # Put images here
│   └── *.jpg
├── output/             # Results appear here
│   ├── consolidated_results.json
│   ├── corrected_transcripts.json
│   └── ...
└── elie.sif           # Container file
```

### Binding Paths

The `--bind` flag maps directories:

```bash
# Map local path to container path
apptainer run --bind /local/path:/container/path elie.sif

# Multiple bindings
apptainer run \
  --bind /scratch/input:/app/data/input \
  --bind /scratch/output:/app/data/output \
  --bind /home/$USER/models:/app/models \
  elie.sif mli
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# View live output
tail -f elie-<jobid>.out

# Check errors
tail -f elie-<jobid>.err

# Cancel job
scancel <jobid>
```

## Tips

1. **Build on login node** - Container building doesn't need compute resources
2. **Use /scratch** - Much faster than home directory
3. **Test interactively first** - Debug before submitting batch jobs
4. **Use job arrays** - Process multiple datasets efficiently
5. **Check module names** - Some systems use `singularity` instead of `apptainer`

## Troubleshooting

**"command not found: apptainer"**
```bash
module load apptainer  # or module load singularity
```

**"No space left on device"**
```bash
# Use /scratch instead of home directory
# Or set temp directory: export APPTAINER_TMPDIR=/scratch/$USER/tmp
```

**"Bind point does not exist in container"**
```bash
# Make sure source path exists and is absolute
ls /scratch/data  # Check it exists first
```

**Pipeline fails on HPC but works locally**
- Check memory limits: increase `--mem` in SLURM
- Check input paths: use absolute paths
- Check modules: some HPCs need `module load python` or similar

## Getting Help

- Check job output: `cat elie-<jobid>.out`
- Check job errors: `cat elie-<jobid>.err`
- Test interactively: `salloc` then run commands manually
- Apptainer help: `apptainer run elie.sif` (shows usage)
- Container help: `apptainer run-help elie.sif`

## Converting from Docker

If you have the Docker image locally:

```bash
# Option 1: Build from definition file (recommended)
apptainer build elie.sif elie.def

# Option 2: Convert existing Docker image
apptainer build elie.sif docker-daemon://label-pipeline:latest
```

The Apptainer container includes everything from the Docker setup.
