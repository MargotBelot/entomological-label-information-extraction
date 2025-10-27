# Pipeline Docker Usage Guide

This directory contains a **consolidated Docker setup** with a single multi-stage Dockerfile and unified docker-compose configuration.

## Quick Start

### Docker

**Multi-Label Image Pipeline (MLI)** - Process full specimen photos:
```bash
cd pipelines
docker-compose --profile mli up
```

**Single-Label Image Pipeline (SLI)** - Process pre-cropped labels:
```bash
cd pipelines
docker-compose --profile sli up
```

### Apptainer/Singularity (HPC Environments)

**Build the container:**
```bash
cd pipelines
apptainer build elie.sif elie.def
```

**Run MLI pipeline:**
```bash
apptainer run --bind $(pwd)/data:/app/data elie.sif mli
```

**Run SLI pipeline:**
```bash
apptainer run --bind $(pwd)/data:/app/data elie.sif sli
```

**Run individual commands:**
```bash
# Detection only
apptainer run --app detection --bind $(pwd)/data:/app/data elie.sif \
  -j /app/data/MLI/input -o /app/data/MLI/output

# Or use exec for any script
apptainer exec --bind $(pwd)/data:/app/data elie.sif \
  python /app/scripts/processing/rotation.py -i /app/data/input -o /app/data/output
```

**For complete HPC documentation**, see `HPC_QUICKSTART.md` for:
- SLURM job scripts
- Job arrays for batch processing
- Resource recommendations
- Troubleshooting on HPC systems

### Run Individual Services

Process one stage at a time with custom paths:

```bash
cd pipelines

# Detection/Segmentation only
docker-compose up segmentation

# Rotation correction only  
docker-compose up rotation

# Classification services
docker-compose up empty_labels          # Empty/not-empty classification
docker-compose up classification_nuri   # Identifier vs description
docker-compose up classification_hp     # Handwritten vs printed

# OCR processing
docker-compose up tesseract

# Post-processing
docker-compose up postprocessing
```

### Custom Paths

Override default paths with environment variables:

```bash
# Custom MLI pipeline
docker-compose run --rm \
  -e INPUT_DIR=/custom/input \
  -e OUTPUT_DIR=/custom/output \
  segmentation

# Custom rotation with specific paths
docker-compose run --rm rotation \
  python3 scripts/processing/rotation.py \
  -i data/custom/input -o data/custom/output
```

## Architecture

### Single Dockerfile (`Dockerfile`)

All services are built from one consolidated Dockerfile with multiple stages:
- **base**: Common dependencies (Python, OpenCV, system libraries)
- **segmentation**: Detection model + dependencies
- **rotation**: Rotation correction model
- **classification**: Classification models
- **tesseract**: Tesseract OCR + dependencies
- **empty_labels**: Empty label detection
- **postprocessing**: Post-processing + NLTK

**Benefits:**
- Shared base layer = faster builds
- Single source of truth
- Easier maintenance
- Smaller total disk usage

### Apptainer Definition (`elie.def`)

Single-file container definition for HPC environments:
- Builds from the same Python base as Docker
- Includes all pipeline dependencies
- Provides built-in apps for each pipeline stage
- Full MLI and SLI pipelines included

**Why Apptainer for HPC:**
- No root privileges required to run
- Better security model for shared systems
- Native HPC scheduler integration
- Compatible with Singularity
- Can be built from Docker images

### Unified Docker Compose (`docker-compose.yml`)

One docker-compose file manages all pipelines using **profiles**:

**Profiles:**
- `mli` - Full Multi-Label Image pipeline (detection → classification → OCR → post-processing)
- `sli` - Full Single-Label Image pipeline (classification → rotation → OCR → post-processing)
- `standalone` - Individual services without dependencies

**Service naming convention:**
- Base services: `segmentation`, `rotation`, `tesseract`, etc.
- Pipeline-specific: `*_mli` or `*_sli` suffix (e.g., `empty_labels_mli`, `rotation_sli`)

## Data Structure

```
data/
├── MLI/
│   ├── input/          # Put full specimen photos here
│   └── output/         # Results appear here
└── SLI/
    ├── input/          # Put pre-cropped labels here
    └── output/         # Results appear here
```

## Building Images

Build all stages at once:
```bash
cd pipelines

# Build specific stage
docker-compose build segmentation
docker-compose build tesseract

# Build all services
docker-compose build
```

## Resource Limits

Default resource allocations (adjustable in `docker-compose.yml`):
- **Detection**: 6GB RAM, 4 CPUs
- **Classification**: 3GB RAM, 2 CPUs
- **Tesseract**: 4GB RAM, 3 CPUs
- **Rotation**: 3GB RAM, 2 CPUs
- **Post-processing**: 2GB RAM, 2 CPUs

## Troubleshooting

**Build fails:**
```bash
# Clear Docker cache
docker system prune -a
docker-compose build --no-cache
```

**Service dependencies not met:**
```bash
# For MLI pipeline, ensure detection completes first
# For SLI pipeline, services run in sequence automatically
# Check healthchecks in docker-compose.yml
```

**Out of memory:**
```bash
# Increase Docker Desktop memory allocation
# Or adjust resource limits in docker-compose.yml
```

## HPC Usage Examples

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=elie-mli
#SBATCH --output=elie-%j.out
#SBATCH --error=elie-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Load Apptainer module (if needed)
module load apptainer

# Set paths
DATA_DIR=/scratch/$USER/elie_data
CONTAINER=/path/to/elie.sif

# Run MLI pipeline
apptainer run --bind $DATA_DIR:/app/data $CONTAINER mli

# Or run SLI pipeline
# apptainer run --bind $DATA_DIR:/app/data $CONTAINER sli
```

### Job Array for Multiple Datasets

```bash
#!/bin/bash
#SBATCH --job-name=elie-array
#SBATCH --array=1-10
#SBATCH --output=elie-%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem=16G

module load apptainer

# Process different input directories
INPUT=/scratch/datasets/batch_${SLURM_ARRAY_TASK_ID}
OUTPUT=/scratch/results/batch_${SLURM_ARRAY_TASK_ID}

INPUT_DIR=$INPUT OUTPUT_DIR=$OUTPUT \
  apptainer run --bind /scratch:/app/data elie.sif mli
```

### Interactive Session

```bash
# Start interactive session
salloc --mem=16G --cpus-per-task=4 --time=2:00:00

# Run pipeline interactively
apptainer run --bind $(pwd)/data:/app/data elie.sif mli

# Or run individual commands
apptainer exec --bind $(pwd)/data:/app/data elie.sif \
  python /app/scripts/processing/detection.py -j /app/data/input -o /app/data/output
```

## Migration from Old Structure

If you were using the old separate `.dockerfile` files:

**Old:**
```bash
docker-compose -f multi-label-docker-compose.yaml up
docker-compose -f single-label-docker-compose.yaml up
```

**New:**
```bash
docker-compose --profile mli up
docker-compose --profile sli up
```

All functionality is preserved - just simpler!
