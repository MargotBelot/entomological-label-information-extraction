# User Guide

🐛 **Complete guide to extracting text from insect specimen labels using AI**

This guide shows you how to use the entomological label extraction tool with practical examples and visual explanations.

## What This Tool Does (For Beginners)

**If you're new to this:** This tool automatically reads text from insect specimen labels and converts it into spreadsheet data.


### **Two Ways to Use This Tool:**

🐳 **Docker (Beginner-Friendly):** Automated, one-click processing
🐍 **Python Scripts (Advanced):** Step-by-step control for custom workflows

### **Which Pipeline Should I Use?**

```
🤔 What type of images do you have?
┌──────────────────────────────────────────────────┐
│                                                  │
│ 📸 Full specimen photos                        │
│ (insects with multiple labels visible)         │
│                    ↓                          │
│        🐳 Use MULTI-LABEL PIPELINE            │
│     docker compose -f multi-label-docker-     │
│            compose.yaml up --build             │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                                                  │
│ 🏷️ Individual label images                      │
│ (already cropped, one label per image)         │
│                    ↓                          │
│        🐳 Use SINGLE-LABEL PIPELINE           │
│     docker compose -f single-label-docker-     │
│            compose.yaml up --build             │
└──────────────────────────────────────────────────┘
```

**Still not sure?** Start with the **Multi-Label Pipeline** – it works for both types!

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

✅ **Required format:** .jpg, .jpeg only

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

## Docker Pipeline Usage

**Prerequisites:**
- Docker Desktop installed and running
- 8GB+ RAM allocated to Docker
- At least 4GB free disk space

### **Docker Pipeline Options**

#### **1. Multi-Label Pipeline** (`multi-label-docker-compose.yaml`)
**Use this when:** You have full specimen photographs with multiple labels per image

**What it does:**
1. **🔍 Detection** - Finds and extracts labels from specimen photos
2. **🏷️ Empty Filter** - Removes blank labels
3. **🎯 Identifier Filter** - Separates catalog numbers from descriptive labels
4. **✍️ Text Type Classification** - Sorts handwritten vs. printed labels
5. **📝 OCR Processing** - Extracts text from printed labels
6. **⚙️ Post-processing** - Creates final structured data

**Input:** Full specimen photos in `data/MLI/input/`
**Final Output:** `data/MLI/output/final_processed_data.csv`

#### **2. Single-Label Pipeline** (`single-label-docker-compose.yaml`)
**Use this when:** You have pre-cropped individual label images

**What it does:**
1. **🏷️ Empty Filter** - Removes blank labels
2. **🎯 Identifier Filter** - Separates catalog numbers from descriptive labels  
3. **✍️ Text Type Classification** - Sorts handwritten vs. printed labels
4. **🔄 Rotation Correction** - Aligns text for better OCR
5. **📝 OCR Processing** - Extracts text from printed labels
6. **⚙️ Post-processing** - Creates final structured data

**Input:** Individual label images in `data/SLI/input/`
**Final Output:** `data/SLI/output/final_processed_data.csv`

### **Running Docker Pipelines**

#### **Quick Start (Multi-Label):**
```bash
# Run with sample data
docker compose -f multi-label-docker-compose.yaml up --build

# Check results
ls data/MLI/output/
head data/MLI/output/final_processed_data.csv
```

#### **Quick Start (Single-Label):**
```bash
# Run with sample data
docker compose -f single-label-docker-compose.yaml up --build

# Check results
ls data/SLI/output/
head data/SLI/output/final_processed_data.csv
```

#### **Using Your Own Images:**

**For Multi-Label Pipeline (Full Specimens):**
```bash
# 1. Prepare your data
mkdir -p data/MLI/input
cp your_specimen_photos/*.jpg data/MLI/input/

# 2. Clear previous results
rm -rf data/MLI/output/*

# 3. Run the complete pipeline
docker compose -f multi-label-docker-compose.yaml up --build

# 4. View results
echo "📊 Final Results:"
ls data/MLI/output/
echo "\n📈 Summary:"
wc -l data/MLI/output/final_processed_data.csv
```

**For Single-Label Pipeline (Pre-cropped Labels):**
```bash
# 1. Prepare your data
mkdir -p data/SLI/input
cp your_label_images/*.jpg data/SLI/input/

# 2. Clear previous results
rm -rf data/SLI/output/*

# 3. Run the complete pipeline
docker compose -f single-label-docker-compose.yaml up --build

# 4. View results
echo "📊 Final Results:"
ls data/SLI/output/
echo "\n📈 Summary:"
wc -l data/SLI/output/final_processed_data.csv
```

### **Visual Pipeline Flow**

#### **Multi-Label Pipeline:**
```
📸 Specimen Photos (data/MLI/input/)
    ↓
🔍 Detection → 📋 input_predictions.csv + 🖼️ input_cropped/
    ↓
🏷️ Empty Filter → 📁 not_empty/ + 🗑️ empty/
    ↓
🎯 Identifier Filter → 📁 identifier/ + 📁 not_identifier/
    ↓
✍️ Text Type Filter → 📁 handwritten/ + 📁 printed/
    ↓
📝 OCR Processing → 📄 ocr_preprocessed.json
    ↓
⚙️ Post-processing → 📊 final_processed_data.csv
```

#### **Single-Label Pipeline:**
```
🏷️ Label Images (data/SLI/input/)
    ↓
🏷️ Empty Filter → 📁 not_empty/ + 🗑️ empty/
    ↓
🎯 Identifier Filter → 📁 identifier/ + 📁 not_identifier/
    ↓
✍️ Text Type Filter → 📁 handwritten/ + 📁 printed/
    ↓
🔄 Rotation Correction → 📁 rotated/
    ↓
📝 OCR Processing → 📄 ocr_preprocessed.json
    ↓
⚙️ Post-processing → 📊 final_processed_data.csv
```

### **Docker Pipeline Control**

**Run specific steps only:**
```bash
# Run only detection step
docker compose -f multi-label-docker-compose.yaml up detection

# Run detection + classification (stop before OCR)
docker compose -f multi-label-docker-compose.yaml up detection handwriten_printed_classifier

# Continue from OCR step
docker compose -f multi-label-docker-compose.yaml up tesseract postprocessing
```

**Monitor progress:**
```bash
# Watch real-time logs
docker compose -f multi-label-docker-compose.yaml up --build | tee pipeline.log

# Check specific service logs
docker compose logs detection
docker compose logs tesseract
```

**Restart and cleanup:**
```bash
# Stop all containers
docker compose -f multi-label-docker-compose.yaml down

# Clean restart
docker compose -f multi-label-docker-compose.yaml down --remove-orphans
docker compose -f multi-label-docker-compose.yaml up --build
```

### **Understanding Your Results**

#### **Output Directory Structure:**
```
data/MLI/output/ (or data/SLI/output/)
├── 📊 final_processed_data.csv    ← 🎯 MAIN RESULT (structured data)
├── 📋 input_predictions.csv       ← Detection coordinates
├── 📄 ocr_preprocessed.json       ← Raw text extraction
├── 📁 input_cropped/              ← Individual label images
│   ├── specimen1_1.jpg
│   ├── specimen1_2.jpg
│   └── specimen2_1.jpg
├── 📁 not_empty/                  ← Labels with content
├── 📁 identifier/                 ← Catalog numbers, IDs
├── 📁 not_identifier/             ← Descriptive labels
├── 📁 handwritten/                ← Hand-written text
├── 📁 printed/                    ← Machine-printed text
└── 📁 rotated/                    ← Text-aligned images (SLI only)
```

#### **What Each File Contains:**

**📊 `final_processed_data.csv`** - **This is your main result!**
```csv
specimen_id,species,location,date,collector,method
CASENT123456,"Lepidoptera sp.","Texas, USA","1995-06-15","J. Smith","AI_extraction"
```

**📋 `input_predictions.csv`** - Detection details
```csv
filename,class,score,xmin,ymin,xmax,ymax
specimen1.jpg,label,0.95,100,150,300,250
```

**📄 `ocr_preprocessed.json`** - Raw OCR text
```json
{
  "ID": "specimen1_1.jpg",
  "text": "Lepidoptera\nTexas, USA\nJune 15, 1995"
}
```

## Step-by-Step Pipeline (Python Scripts)

**For users who want more control or to run individual steps:**

### **Multi-Label Workflow (Specimen Photos)**

#### **Step 1: Label Detection**
```bash
# Find and extract labels from specimen photos
python3 scripts/processing/detection.py -j data/MLI/input -o data/MLI/output

# What this creates:
# - input_predictions.csv (detection results)
# - input_cropped/ (individual label images)
```

#### **Step 2: Empty Label Analysis**
```bash
# Filter out blank or illegible labels
python3 scripts/processing/analysis.py -o data/MLI/output -i data/MLI/output/input_cropped

# What this creates:
# - not_empty/ (labels with content)
# - empty/ (blank labels, filtered out)
```

#### **Step 3: Identifier Classification**
```bash
# Separate specimen IDs from descriptive labels
python3 scripts/processing/classifiers.py -m 1 -j data/MLI/output/not_empty -o data/MLI/output

# What this creates:
# - identifier/ (catalog numbers, specimen IDs)
# - not_identifier/ (locality, date, collector info)
```

#### **Step 4: Handwritten/Printed Classification**
```bash
# Categorize labels by text type
python3 scripts/processing/classifiers.py -m 2 -j data/MLI/output/not_identifier -o data/MLI/output

# What this creates:
# - handwritten/ (hand-written labels)
# - printed/ (machine-printed labels, ready for OCR)
```

#### **Step 5: OCR Text Extraction**
```bash
# Extract text from printed labels
python3 scripts/processing/tesseract.py -d data/MLI/output/printed -o data/MLI/output

# What this creates:
# - ocr_preprocessed.json (extracted text with metadata)
```

#### **Step 6: Post-processing**
```bash
# Clean and structure the extracted data
python3 scripts/postprocessing/process.py -j data/MLI/output/ocr_preprocessed.json -o data/MLI/output

# What this creates:
# - final_processed_data.csv (clean, structured specimen data)
```

### **Single-Label Workflow (Pre-cropped Labels)**

#### **Step 1: Empty Label Analysis**
```bash
# Filter out blank labels
python3 scripts/processing/analysis.py -o data/SLI/output -i data/SLI/input

# Creates: not_empty/, empty/
```

#### **Step 2: Identifier Classification**
```bash
# Separate IDs from descriptive content
python3 scripts/processing/classifiers.py -m 1 -j data/SLI/output/not_empty -o data/SLI/output

# Creates: identifier/, not_identifier/
```

#### **Step 3: Handwritten/Printed Classification**
```bash
# Sort by text type
python3 scripts/processing/classifiers.py -m 2 -j data/SLI/output/not_identifier -o data/SLI/output

# Creates: handwritten/, printed/
```

#### **Step 4: Rotation Correction**
```bash
# Align text horizontally for better OCR
python3 scripts/processing/rotation.py -o data/SLI/output/rotated -i data/SLI/output/printed

# Creates: rotated/ (text-aligned images)
```

#### **Step 5: OCR Text Extraction**
```bash
# Extract text from aligned labels
python3 scripts/processing/tesseract.py -d data/SLI/output/rotated -o data/SLI/output

# Creates: ocr_preprocessed.json
```

#### **Step 6: Post-processing**
```bash
# Create final structured dataset
python3 scripts/postprocessing/process.py -j data/SLI/output/ocr_preprocessed.json -o data/SLI/output

# Creates: final_processed_data.csv
```

### **Script Options Reference**

#### **Detection Script Options:**
```bash
python3 scripts/processing/detection.py [OPTIONS]
  -j, --input-dir DIR      # Directory with specimen photos
  -i, --input-image FILE   # Single specimen photo  
  -o, --output-dir DIR     # Where to save results
  --confidence FLOAT       # Detection threshold (0.3-0.9, default: 0.5)
  --batch-size INT         # Processing batch size (default: 16)
  --device TEXT            # 'cpu', 'cuda', or 'auto'
```

#### **Classification Script Options:**
```bash
python3 scripts/processing/classifiers.py [OPTIONS]
  -m, --model INT          # 1=identifier, 2=handwritten/printed, 3=multi/single
  -j, --jpg_dir DIR        # Input directory with label images
  -o, --out_dir DIR        # Output directory
```

#### **OCR Script Options:**
```bash
python3 scripts/processing/tesseract.py [OPTIONS]
  -d, --dir DIR            # Directory with label images
  -o, --outdir DIR         # Output directory for JSON results
  -v, --verbose            # Show detailed processing info
  -t, --thresholding INT   # 1=Otsu, 2=Adaptive, 3=Gaussian (default: 1)
```

#### **Rotation Script Options:**
```bash
python3 scripts/processing/rotation.py [OPTIONS]
  -i, --input_dir DIR      # Directory with images to rotate
  -o, --output_dir DIR     # Directory for rotated images
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
