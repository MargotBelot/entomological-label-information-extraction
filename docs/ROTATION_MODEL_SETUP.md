# Rotation Model Setup and Troubleshooting

This guide covers setup, troubleshooting, and training for the rotation correction model.

## Quick Setup

### 1. Download Pre-trained Model

```bash
# Navigate to your project directory
cd entomological-label-information-extraction

# Create models directory if it doesn't exist
mkdir -p models

# Download the pre-trained rotation model
# Replace with your actual model URL or location
wget https://your-model-repository.com/rotation_model.h5 -O models/rotation_model.h5

# Or copy from shared location
cp /path/to/shared/rotation_model.h5 models/
```

### 2. Verify Model Installation

```bash
# Check that the model exists
ls -lh models/rotation_model.h5

# Test the rotation script
python scripts/processing/rotation.py \
  -i data/SLI/input \
  -o data/SLI/output/test_rotation
```

## Model Search Order

The rotation script searches for models in this order:

1. `models/rotation_model.h5` (recommended)
2. `models/label_rotation_model.h5` (alternative)
3. `models/rotation_classifier.h5` (alternative)

If none are found, the script will exit with an error message showing all searched paths.

## Troubleshooting

### Error: "Rotation model not found"

**Symptoms:**
```
Error: Rotation model not found. Tried:
  - /path/to/project/models/rotation_model.h5
  - /path/to/project/models/label_rotation_model.h5
  - /path/to/project/models/rotation_classifier.h5
Please ensure the rotation model is available in the models directory.
```

**Solutions:**

1. **Download the model** (see Quick Setup above)

2. **Check model file permissions:**
   ```bash
   chmod 644 models/rotation_model.h5
   ```

3. **Verify project structure:**
   ```bash
   # From project root
   ls -la models/
   ```

4. **Use alternative model name:**
   ```bash
   # If you have a model with a different name
   cp your_model.h5 models/rotation_model.h5
   ```

### Error: "Model loading failed"

**Symptoms:**
```
Warning: Standard model loading failed: ...
Attempting to load with custom objects...
Error: Could not load model with either method.
```

**Solutions:**

1. **Check TensorFlow version compatibility:**
   ```bash
   conda activate entomological-label
   python -c "import tensorflow as tf; print(tf.__version__)"
   # Should be 2.15.0 or compatible
   ```

2. **Verify model file integrity:**
   ```bash
   # Check file is not corrupted
   file models/rotation_model.h5
   # Should show: HDF5 data file
   ```

3. **Re-download the model:**
   ```bash
   rm models/rotation_model.h5
   # Download again from source
   ```

### Error: "Legacy optimizer not supported"

**Symptoms:**
```
Warning: Compilation with legacy optimizer failed: 
`keras.optimizers.legacy` is not supported in Keras 3.
Retrying with standard Adam optimizer...
```

**This is normal!** The script automatically handles this by:
1. Trying legacy Adam optimizer first
2. Falling back to standard Adam if that fails
3. Processing continues normally

**No action needed** - the fallback mechanism is working correctly.

### Pipeline skips rotation or uses unrotated images

**Symptoms:**
```
⚠️  Both rotation methods failed, using fallback (copying original images)
Note: OCR will proceed with unrotated images (may be less accurate)
```

**Solutions:**

1. **Ensure rotation model exists:**
   ```bash
   ls -la models/rotation_model.h5
   ```

2. **Check rotation script output:**
   ```bash
   python scripts/processing/rotation.py -i test_input -o test_output
   # Look for any error messages
   ```

3. **Verify image format:**
   ```bash
   # Rotation only works with .jpg, .jpeg, .tiff, .tif
   file data/input/*.jpg
   ```

4. **Check Python path:**
   ```bash
   # Ensure label_processing module is accessible
   python -c "from label_processing.label_rotation import predict_angles"
   ```

## Training Your Own Model

If you need to train a custom rotation model:

### Data Preparation

1. Collect labeled images at different rotations (0°, 90°, 180°, 270°)
2. Organize in directory structure:
   ```
   training_data/
   ├── 0/      # Images at 0 degrees
   ├── 90/     # Images at 90 degrees
   ├── 180/    # Images at 180 degrees
   └── 270/    # Images at 270 degrees
   ```

### Training Script

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model architecture (example)
def create_rotation_model():
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 classes: 0, 90, 180, 270
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train
model = create_rotation_model()
# ... training code ...
model.save('models/rotation_model.h5')
```

### Model Requirements

Your rotation model must:
- Accept input shape: `(224, 224, 3)` (RGB images resized to 224x224)
- Output 4 classes representing rotation angles:
  - Class 0: 0° (no rotation)
  - Class 1: 90° clockwise
  - Class 2: 180°
  - Class 3: 270° clockwise
- Be saved in HDF5 format (`.h5` extension)

## Advanced Configuration

### Custom Model Path

Set environment variable to use a model from a different location:

```bash
export ROTATION_MODEL_PATH="/custom/path/to/rotation_model.h5"
```

Then modify `scripts/processing/rotation.py` to check this variable.

### Skip Rotation Step

If you don't need rotation correction:

**Option 1: Use already rotated images**
- Manually rotate images before processing
- Place in input directory

**Option 2: Modify pipeline script**
```bash
# Comment out rotation step in pipeline scripts
# tools/pipelines/run_mli_pipeline_conda.sh
# tools/pipelines/run_sli_pipeline_conda.sh
```

**Option 3: Process without rotation**
```bash
# Run OCR directly on unrotated images
python scripts/processing/tesseract.py -d data/input -o data/output
```

## Performance Notes

### Rotation Speed

- **With GPU:** ~0.1-0.2 seconds per image
- **With CPU:** ~0.5-2 seconds per image
- **Batch size:** Processes all images in directory at once

### Accuracy

The pre-trained model typically achieves:
- **Overall accuracy:** >95% on test data
- **Most common errors:** Confusion between 0° and 180° for symmetric labels

### When Rotation Fails

If rotation produces poor results:
1. Check image quality (blur, low contrast affect detection)
2. Verify images contain actual text (rotation uses text orientation)
3. Consider manual pre-processing for problem images

## Testing

### Test Rotation on Sample Images

```bash
# Use debug mode to see predicted angles
cd entomological-label-information-extraction

# Create test directory
mkdir -p test_rotation

# Test rotation
python scripts/processing/rotation.py \
  -i data/SLI/input \
  -o test_rotation

# Check results
ls -la test_rotation/
```

### Validate Rotation Accuracy

```python
# Simple accuracy check
import cv2
import numpy as np

# Load original and rotated images
original = cv2.imread('original.jpg')
rotated = cv2.imread('rotated.jpg')

# Visual inspection or automated checks
print(f"Original shape: {original.shape}")
print(f"Rotated shape: {rotated.shape}")
```

## Support

If you continue to have issues:

1. Check the project [README.md](../README.md) for general setup
2. Review [ADVANCED_CONFIG.md](ADVANCED_CONFIG.md) for environment variables
3. Open an issue on the project repository with:
   - Error message
   - Output of `python scripts/processing/rotation.py`
   - TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`
   - Operating system and Python version

## See Also

- [ADVANCED_CONFIG.md](ADVANCED_CONFIG.md) - Advanced configuration options
- [README.md](../README.md) - Main project documentation
- `label_processing/label_rotation.py` - Rotation module source code
