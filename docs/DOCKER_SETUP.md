# Docker Pipeline Setup

This document explains the Docker-based pipeline setup for entomological label processing. Docker provides the **easiest way** to run the complete pipeline with zero configuration.

## Pipeline Types

**Choose the right pipeline for your data:**

### Multi-Label Pipeline
- **Input**: Full specimen images in `data/MLI/input/`
- **Process**: Detection → Classification → OCR → Post-processing
- **Output**: `data/MLI/output/consolidated_results.json`

### Single-Label Pipeline  
- **Input**: Pre-cropped label images in `data/SLI/input/`
- **Process**: Classification → Rotation → OCR → Post-processing
- **Output**: `data/SLI/output/consolidated_results.json`

## Quick Start

### Option 1: Use the Main Launcher (Recommended)
```bash
./run-pipeline.sh
```

### Option 2: Run Specific Pipelines Directly
```bash
# Multi-label pipeline (full specimen images → detected labels)
./scripts/docker/start-multi-label-pipeline.sh

# Single-label pipeline (pre-cropped images → processed text)  
./scripts/docker/start-single-label-pipeline.sh
```

### Option 3: Validate Setup
```bash
./scripts/docker/validate-docker-setup.sh
```

## Features

### Automatic Docker Management
-  Automatically starts Docker Desktop if not running
-  **Automatically creates input/output directories**
-  Cleans up previous pipeline runs
-  Handles container lifecycle management
-  Removes orphaned containers and processes

### Health Monitoring
-  Service health checks
-  Pipeline progress monitoring
-  Error detection and reporting
-  Automatic cleanup on failure

### Consolidated Output
-  Single JSON file with all pipeline results
-  Per-file entity linking across all processing stages
-  Complete processing history and metadata
-  Summary statistics

## Directory Structure

```
data/
├── MLI/                    # Multi-Label Input/Output
│   ├── input/             # Full specimen images
│   └── output/            # Detection + processing results
└── SLI/                   # Single-Label Input/Output
    ├── input/             # Pre-cropped label images
    └── output/            # Classification + processing results

scripts/
├── docker/                # Docker management scripts
│   ├── start-multi-label-pipeline.sh
│   ├── start-single-label-pipeline.sh
│   └── validate-docker-setup.sh
├── processing/            # Core processing scripts
└── postprocessing/        # Post-processing scripts

pipelines/                 # Docker configuration
├── *.dockerfile          # Service definitions
├── multi-label-docker-compose.yaml    # Multi-label pipeline
├── single-label-docker-compose.yaml   # Single-label pipeline  
└── requirements/          # Python dependencies

docs/                      # Documentation
├── DOCKER_SETUP.md        # Docker setup guide
├── TECHNICAL_GUIDE.md     # Technical documentation
└── USER_GUIDE.md          # User manual
```

## Docker Compose Services

### Multi-Label Pipeline Services
1. **detection** - Detect and crop labels from specimen images
2. **empty_not_empty_classifier** - Filter empty labels
3. **nuri_notnuri_classifier** - Classify identifier vs description
4. **handwritten_printed_classifier** - Classify text type
5. **tesseract** - OCR text extraction
6. **postprocessing** - Clean text and consolidate results

### Single-Label Pipeline Services
1. **empty_not_empty_classifier** - Filter empty labels
2. **nuri_notnuri_classifier** - Classify identifier vs description
3. **handwritten_printed_classifier** - Classify text type
4. **rotator** - Correct text orientation
5. **tesseract** - OCR text extraction
6. **postprocessing** - Clean text and consolidate results

## Environment Setup

### Requirements

**All Platforms:**
- At least 4GB RAM available for Docker
- 10GB+ free disk space
- Internet connection for initial image downloads

**Platform-Specific Docker Installation:**
- **macOS**: Docker Desktop from https://docker.com
- **Windows**: Docker Desktop from https://docker.com (requires WSL2)
- **Linux**: 
  - Docker Engine: `sudo apt install docker.io` (Ubuntu/Debian)
  - Or Docker Desktop from https://docker.com
  - Ensure user is in docker group: `sudo usermod -aG docker $USER`

### First Run
The first pipeline run will be slower (10-15 minutes) as Docker images are built. Subsequent runs will be much faster (2-5 minutes).

## Troubleshooting

### Common Issues

**Docker not installed or not found:**
- Install Docker for your platform (see Requirements above)
- On Linux, make sure Docker service is running: `sudo systemctl start docker`
- On Linux, ensure user permissions: `sudo usermod -aG docker $USER` (then logout/login)

**Permission denied on scripts:**
```bash
# Make scripts executable
chmod +x run-pipeline.sh
chmod +x scripts/docker/*.sh
```

**Docker daemon not running:**
- **macOS/Windows**: Start Docker Desktop application
- **Linux**: `sudo systemctl start docker`

**Out of disk space:**
```bash
# Clean Docker cache and unused images
docker system prune -f
docker image prune -f
```

**Build failures:**
```bash
# First, validate your setup
./scripts/docker/validate-docker-setup.sh

# Check specific service logs
docker-compose -f pipelines/multi-label-docker-compose.yaml logs [service-name]
```

**WSL2 issues (Windows):**
- Ensure WSL2 is installed and updated
- Enable WSL2 integration in Docker Desktop settings
- Run pipelines from within WSL2 terminal, not Windows Command Prompt

### Manual Docker Commands

If you need to run individual services manually:

```bash
# Build specific service
docker-compose -f pipelines/multi-label-docker-compose.yaml build detection

# Run specific service
docker-compose -f pipelines/multi-label-docker-compose.yaml up detection

# View logs
docker-compose -f pipelines/multi-label-docker-compose.yaml logs -f

# Clean up
docker-compose -f pipelines/multi-label-docker-compose.yaml down --remove-orphans
```

## Output Format

The consolidated results JSON contains per-file processing information:

```json
[
  {
    "filename": "specimen1_label_1.jpg",
    "detection": {
      "coordinates": [100, 150, 300, 250],
      "confidence": 0.95
    },
    "classification": {
      "empty": false,
      "identifier": false, 
      "handwritten": false,
      "printed": true
    },
    "rotation": {
      "angle": 90,
      "corrected": true
    },
    "ocr": {
      "method": "tesseract",
      "raw_text": "Lepidoptera\\nTexas, USA\\nJune 15, 1995",
      "confidence": 0.87
    },
    "postprocessing": {
      "original_text": "Lepidoptera\\nTexas, USA\\nJune 15, 1995", 
      "cleaned_text": "Lepidoptera Texas USA June 15 1995",
      "corrected": true,
      "plausible": true,
      "category": "descriptive"
    }
  }
]
```

## Support

For issues or questions about the Docker setup, check:
1. Run the validation script: `./scripts/docker/validate-docker-setup.sh`
2. Check Docker Desktop is running and has sufficient resources
3. Review Docker logs for specific error messages
