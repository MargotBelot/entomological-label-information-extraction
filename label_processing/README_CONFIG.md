# Configuration System

This directory contains the centralized configuration system for cross-platform path management.

## Files

- `config.py` - Main configuration module that handles:
  - Automatic project root detection
  - Platform-independent path resolution
  - Model path management
  - Environment variable overrides

## Usage

```python
from label_processing.config import get_model_path, config

# Get specific model paths
detection_model = get_model_path("detection")
classifier_model = get_model_path("identifier")

# Access configuration object directly
project_root = config.project_root
models_dir = config.models_dir
```

## Environment Variables

You can override default paths using environment variables:

```bash
export ENTOMOLOGICAL_PROJECT_ROOT="/custom/project/path"
export ENTOMOLOGICAL_MODELS_DIR="/custom/models/path"
export ENTOMOLOGICAL_DETECTION_MODEL_PATH="/custom/model.pth"
```

## Cross-Platform Compatibility

The configuration system automatically handles path differences between:
- Linux (`/home/user/...`)
- macOS (`/Users/user/...`) 
- Windows (`C:\Users\user\...`)

All scripts import from this centralized location to ensure consistent path handling across the entire project.
