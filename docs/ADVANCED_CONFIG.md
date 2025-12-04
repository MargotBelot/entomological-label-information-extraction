# Advanced Configuration

This document covers advanced configuration options for the Entomological Label Information Extraction pipeline.

## Environment Variables

The pipeline supports several environment variables to customize paths and behavior:

### Project Path Overrides

```bash
# Override the project root directory
export ENTOMOLOGICAL_PROJECT_ROOT="/path/to/project"

# Override the models directory location
export ENTOMOLOGICAL_MODELS_DIR="/path/to/models"

# Override the detection model path specifically
export ENTOMOLOGICAL_DETECTION_MODEL_PATH="/path/to/custom/detection_model.pth"
```

**Use Cases:**
- Running the pipeline from a non-standard location
- Sharing models across multiple project instances
- Using models stored on network drives or external storage

### Pipeline Path Overrides

```bash
# Override input/output directories for pipelines
export INPUT_DIR="/path/to/input"
export OUTPUT_DIR="/path/to/output"
```

**Use Cases:**
- Processing images from external drives
- Writing outputs to specific locations
- Batch processing with custom directory structures

## Model Caching

### Detection Model Caching

The detection script (`scripts/processing/detection.py`) implements an intelligent model caching mechanism to speed up repeated runs.

**How it works:**
1. First load: Model is loaded from disk (~10-30 seconds)
2. Cache created: Model state is saved to `~/.entomological_cache/`
3. Subsequent loads: Model loads from cache (~2-5 seconds)

**Cache location:**
```
~/.entomological_cache/
└── model_<hash>.pkl
```

**Cache validation:**
- Automatically detects model file changes
- Uses MD5 hash of model file for validation
- Invalidates cache if model is updated

**Disable caching:**
```python
# In detection.py, modify:
predictor = OptimizedPredictLabel(
    path_to_model=model_path,
    classes=["label"],
    threshold=THRESHOLD,
    use_cache=False  # Disable caching
)
```

**Clear cache manually:**
```bash
rm -rf ~/.entomological_cache/
```

## Rotation Model Configuration

### Model Search Paths

The rotation script searches for models in the following order:

1. `models/rotation_model.h5` (primary)
2. `models/label_rotation_model.h5` (alternative)
3. `models/rotation_classifier.h5` (alternative)

### Missing Model Handling

If no rotation model is found, the pipeline will:
1. Print error message with searched paths
2. Exit gracefully with error code 1
3. Suggest downloading or placing the model

**Downloading the rotation model:**
```bash
# Download from your model repository
wget https://your-repo.com/models/rotation_model.h5 -O models/rotation_model.h5

# Or train your own rotation model
# See: docs/MODEL_TRAINING.md
```

## Conda Environment Customization

### Using Custom Environment Name

If you want to use a different conda environment name:

1. Edit `environment.yml`:
   ```yaml
   name: your-custom-name  # Change this line
   ```

2. Update pipeline scripts:
   ```bash
   # In tools/pipelines/run_mli_pipeline_conda.sh
   # and tools/pipelines/run_sli_pipeline_conda.sh
   conda activate your-custom-name
   ```

3. Create environment:
   ```bash
   conda env create -f environment.yml
   ```

## Docker Configuration

### Custom Port Mapping

By default, Docker containers use standard ports. To customize:

```bash
# In docker-compose.yml, add port mappings:
services:
  segmentation:
    ports:
      - "8080:8080"  # host:container
```

### Custom Volume Mounts

Mount additional directories:

```yaml
services:
  segmentation:
    volumes:
      - ${PWD}/data:/app/data
      - /external/storage:/app/external  # Additional mount
```

### Memory and CPU Limits

Adjust resource limits based on your hardware:

```yaml
deploy:
  resources:
    limits:
      memory: 8G      # Increase for larger images
      cpus: '6.0'     # Increase for faster processing
    reservations:
      memory: 4G
      cpus: '2.0'
```

## HPC/Apptainer Configuration

### Environment Variables in HPC

When using Apptainer/Singularity on HPC:

```bash
# In your SLURM script:
export APPTAINERENV_ENTOMOLOGICAL_PROJECT_ROOT=/scratch/username/project
export APPTAINERENV_ENTOMOLOGICAL_MODELS_DIR=/scratch/shared/models

apptainer run --bind /scratch/data:/app/data elie.sif mli
```

### Parallel Processing

For HPC batch processing:

```bash
# Process multiple images in parallel across nodes
srun -n 10 --cpus-per-task=4 apptainer run elie.sif mli
```

## Tesseract Configuration

### Custom Tesseract Path

If Tesseract is installed in a non-standard location:

```bash
export TESSERACT_CMD="/custom/path/to/tesseract"
```

### Language Packs

Install additional language packs for OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-fra  # French
sudo apt-get install tesseract-ocr-deu  # German

# macOS
brew install tesseract-lang
```

Then specify in OCR command:
```bash
python scripts/processing/tesseract.py -d input -o output -l eng+fra
```

## Performance Tuning

### Multiprocessing

Enable parallel OCR processing:

```bash
python scripts/processing/tesseract.py \
  -d input \
  -o output \
  -multi  # Enable multiprocessing
```

### Batch Size for Detection

Adjust detection batch size based on available memory:

```bash
python scripts/processing/detection.py \
  -j input \
  -o output \
  --batch-size 4  # Reduce if out of memory
```

### GPU Configuration

If you have a GPU available:

```python
# In detection.py or other PyTorch scripts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Troubleshooting

### Cache Issues

If you experience issues with cached models:

```bash
# Clear all caches
rm -rf ~/.entomological_cache/
rm -rf __pycache__
find . -type d -name "*.egg-info" -exec rm -rf {} +
```

### Path Resolution Issues

Check that paths are correctly resolved:

```python
# Run config validation
python label_processing/config.py

# Expected output shows all paths
```

### Permission Issues

Ensure proper permissions:

```bash
# Make scripts executable
chmod +x tools/pipelines/*.sh

# Fix model file permissions
chmod 644 models/*.h5 models/*.pth
```

## See Also

- [README.md](../README.md) - Main documentation
- [Docker README](../pipelines/README.md) - Docker-specific docs
- [HPC Quickstart](../pipelines/HPC_QUICKSTART.md) - HPC-specific docs
