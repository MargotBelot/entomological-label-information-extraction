# User Guide

This guide shows you how to use the entomological label extraction tool with practical examples.

## Prerequisites

Before starting, make sure you have completed the installation process described in [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

## Quick Start with Sample Data

The repository includes sample data to help you get started immediately:

- `data/MLI/input/` - Multi-label specimen images (2 sample images)
- `data/SLI/input/` - Single-label images (6 sample images)

**Option 1: Docker Pipeline (Recommended)**

```bash
# Multi-label pipeline (includes label detection)
docker compose -f multi-label-docker-compose.yaml up --build
# Results in: data/MLI/output/

# Single-label pipeline (for pre-cropped labels)
docker compose -f single-label-docker-compose.yaml up --build
# Results in: data/SLI/output/
```

**Option 2: Python Scripts**

```bash
# Process the multi-label samples
python3 scripts/processing/detection.py -j data/MLI/input -o data/MLI/output

# Process the single-label samples  
python3 scripts/processing/detection.py -j data/SLI/input -o data/SLI/output
```

## Basic Usage

**Process a folder of images:**

```bash
python3 scripts/processing/detection.py -j /path/to/your/images -o /path/to/output
```

**Process a single image:**

```bash
python3 scripts/processing/detection.py -i specimen.jpg -o results/
```

## Command Options

The main detection script supports these options:

```bash
python3 scripts/processing/detection.py [OPTIONS]
```

**Required (choose one):**
- `-j, --input-dir` - Directory containing specimen images
- `-i, --input-image` - Single image file

**Required:**
- `-o, --output-dir` - Where to save results

**Optional:**
- `--confidence FLOAT` - Detection confidence threshold (default: 0.5)
- `--batch-size INT` - Images processed simultaneously (default: 16)
- `--device TEXT` - Use 'cpu' or 'cuda' (auto-detected)

**Examples:**

```bash
# High confidence, smaller batches
python3 scripts/processing/detection.py -j images/ -o results/ --confidence 0.8 --batch-size 8

# Force CPU usage
python3 scripts/processing/detection.py -j images/ -o results/ --device cpu
```

## Understanding the Output

After processing, you'll find these files in your output directory:

**CSV Files:**

1. `input_predictions.csv` - Label detection results

```
filename,label_count,bbox_coordinates,confidence_scores
specimen1.jpg,2,"[[100,150,300,250],[120,300,280,380]]","[0.95,0.87]"
```

2. `input_text_extraction.csv` - OCR results (if available)

```
filename,label_text,processing_method
specimen1_label_1.jpg,"Lepidoptera Noctuidae","tesseract"
```

**Image Files:**

- `input_cropped/` - Individual label images extracted from specimens

```
input_cropped/
├── specimen1_label_1.jpg
├── specimen1_label_2.jpg
└── specimen2_label_1.jpg
```

## Working with Your Own Images

**Image Requirements:**

✅ **Supported formats:** .jpg, .jpeg, .png, .tiff

✅ **Best quality results:**
- High resolution (300+ DPI)
- Clear, well-lit labels
- Horizontal text orientation
- Contrasting background

✅ **Typical workflow:**
1. Place images in a folder
2. Run the detection script
3. Review results in output folder
4. Manually verify important specimens

**Example workflow:**

```bash
# 1. Organize your images
mkdir my_specimens
cp *.jpg my_specimens/

# 2. Process them
python3 scripts/processing/detection.py -j my_specimens -o results_$(date +%Y%m%d)

# 3. Check the results
ls results_$(date +%Y%m%d)/
head results_$(date +%Y%m%d)/input_predictions.csv
```

## Real-World Examples

**Museum Collection Processing:**

```bash
# Process a large collection with tracking
python3 scripts/processing/detection.py \
    -j /museum/lepidoptera_2024/ \
    -o /results/lepidoptera_batch_$(date +%Y%m%d) \
    --confidence 0.7
```

**Field Research Workflow:**

```bash
# Quick processing of field photos
python3 scripts/processing/detection.py \
    -j ./field_trip_photos \
    -o ./extracted_labels \
    --confidence 0.6
```

**Quality Control Check:**

```bash
# High confidence for critical specimens
python3 scripts/processing/detection.py \
    -j ./type_specimens \
    -o ./verified_labels \
    --confidence 0.9
```

## Batch Processing Tips

**For Large Collections:**

1. **Organize by batches:** Process 100-500 images at a time
2. **Monitor progress:** Check output files periodically
3. **Verify quality:** Review a sample of results before processing more
4. **Save settings:** Document the parameters that work best for your images

**Memory Management:**

```bash
# For limited RAM systems
python3 scripts/processing/detection.py -j images/ -o results/ --batch-size 4

# For high-performance systems
python3 scripts/processing/detection.py -j images/ -o results/ --batch-size 32
```

## Docker Usage

**Prerequisites:**
- Docker Desktop installed and running
- 8GB+ RAM allocated to Docker

**Available Docker Pipelines:**

1. **Multi-label Pipeline** (`multi-label-docker-compose.yaml`)
   - Full pipeline including label detection
   - Input: Specimen photos with multiple labels
   - Includes: Detection → Classification → OCR → Post-processing

2. **Single-label Pipeline** (`single-label-docker-compose.yaml`)
   - For pre-cropped individual labels
   - Input: Individual label images
   - Includes: Classification → OCR → Post-processing

**Using Your Own Images with Docker:**

```bash
# 1. Place your images in the appropriate input folder
cp your_specimens/*.jpg data/MLI/input/     # For multi-label pipeline
cp your_labels/*.jpg data/SLI/input/        # For single-label pipeline

# 2. Run the pipeline
docker compose -f multi-label-docker-compose.yaml up --build

# 3. Results will be in the output folder
ls data/MLI/output/
```

**Docker Troubleshooting:**

```bash
# Stop all containers
docker compose down

# Clean up and rebuild
docker compose down --remove-orphans
docker compose -f multi-label-docker-compose.yaml up --build

# Check container logs
docker compose logs
```

## Using the Python API

For custom workflows, you can use the tool programmatically:

```python
from label_processing.label_detection import PredictLabel

# Initialize the detector
detector = PredictLabel(
    path_to_model="models/detection_model.pth",
    classes=["label"]
)

# Process a single image
results = detector.class_prediction("specimen.jpg")
print(f"Found {len(results)} labels")

# Process multiple images
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in image_list:
    results = detector.class_prediction(img)
    print(f"{img}: {len(results)} labels detected")
```

## Getting Help

**If something goes wrong:**

1. **Check the output folder** - Error messages are often saved there
2. **Try with sample data first** - Verify your installation works
3. **Check image format** - Make sure files are .jpg format
4. **Adjust confidence** - Lower values detect more labels (but more false positives)
5. **See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - For installation and troubleshooting issues

**Common Questions:**

**Q: No labels detected in my images?**
- Try `--confidence 0.3` for more sensitive detection
- Verify labels are clearly visible in the image
- Check that images are not corrupted

**Q: Too many false detections?**
- Increase `--confidence 0.8` for more selective detection
- Ensure images contain only specimen labels

**Q: Processing is very slow?**
- Reduce `--batch-size 4` if you have limited RAM
- Use smaller, lower-resolution images if acceptable

**Q: Can I process non-English labels?**
- The detection works with any language
- OCR text extraction supports multiple languages (see [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md))

**Q: Model loading issues?**
- See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for model loading troubleshooting
- The package includes improved model loading with automatic CPU/GPU fallback

## Next Steps

After extracting labels successfully:

1. **Review the CSV files** - Check detection accuracy
2. **Examine cropped images** - Verify label quality
3. **Use the data** - Import CSV files into your database or analysis software
4. **Customize settings** - Adjust parameters for your specific image types

For advanced usage, model training, and troubleshooting, see [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).
