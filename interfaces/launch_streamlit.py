#!/usr/bin/env python3
"""
Entomological Label Information Extraction - Streamlit Web Interface
Modern web-based launcher with automatic Docker management and real-time processing.
"""

import streamlit as st
import subprocess
import os
import sys
import time
import threading
import queue
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from datetime import datetime
import psutil
import cv2


def _load_consolidated(output_path: Path) -> list:
    """Load label data from entity_master.json or consolidated_results.json.

    Returns a flat list of per-label dicts, each containing source_image,
    label_filename, label_index, category, bbox, rotation_angle, ocr,
    and (when available) entity_extraction.
    """
    # Prefer entity_master.json (has entities embedded)
    master_path = output_path / "entity_master.json"
    if master_path.exists():
        try:
            with open(master_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and data and "labels" in data[0]:
                # Flatten grouped structure into flat label list
                labels = []
                for entry in data:
                    for lbl in entry.get("labels", []):
                        if "source_image" not in lbl:
                            lbl["source_image"] = entry.get("source_image", "")
                        labels.append(lbl)
                return labels
        except Exception:
            pass
    # Fall back to consolidated_results.json
    path = output_path / "consolidated_results.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _group_labels_by_image(consolidated: list) -> dict:
    """Group a flat consolidated list into {source_image: [label, ...]}."""
    grouped: dict = {}
    for entry in consolidated:
        img = entry.get("source_image", "")
        if img:
            grouped.setdefault(img, []).append(entry)
    return grouped


# Page configuration
st.set_page_config(
    page_title="ELIE - Entomological Label Information Extraction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for a clean, professional look ----
st.markdown("""
<style>
/* Tighten vertical spacing */
.block-container { padding-top: 2rem; padding-bottom: 1rem; }

/* Section headers */
h1 { letter-spacing: -0.5px; }
h2 { color: #1565C0 !important; border-bottom: 2px solid #E0E7EF; padding-bottom: 0.3rem; }
h3 { color: #1976D2 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #F5F7FA; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #F5F7FA;
    border: 1px solid #E0E7EF;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Primary button */
button[kind="primary"] {
    background: #1565C0 !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Expanders */
details {
    border: 1px solid #E0E7EF !important;
    border-radius: 6px !important;
}

/* OCR edit text areas */
textarea {
    border: 1px solid #D0D7E0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}

/* Input fields for entity editing */
input[type="text"] {
    border: 1px solid #D0D7E0 !important;
    border-radius: 6px !important;
}

/* Label crop container */
.label-crop-container {
    border: 2px solid #E0E7EF;
    border-radius: 8px;
    padding: 0.5rem;
    background: #FAFBFC;
}

/* OCR edit area container */
.ocr-edit-area {
    border: 2px solid #E0E7EF;
    border-radius: 8px;
    padding: 0.5rem;
    background: #FFFBF0;
}

/* Download buttons full width */
button[data-testid="stDownloadButton"] { width: 100%; }

/* Progress bar */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #1565C0, #42A5F5) !important;
}

/* Dividers */
hr { border-color: #E0E7EF !important; margin: 0.5rem 0 !important; }

/* Hide default footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Clear Streamlit cache to avoid deprecation warnings from cached sessions
if hasattr(st, 'cache_data'):
    st.cache_data.clear()
if hasattr(st, 'cache_resource'):
    st.cache_resource.clear()

class DockerManager:
    """Manages Docker operations and status"""
    
    @staticmethod
    def is_docker_installed():
        """Check if Docker is installed"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def is_docker_running():
        """Check if Docker daemon is running"""
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def start_docker():
        """Attempt to start Docker"""
        try:
            if sys.platform == "darwin":  # macOS
                # Try Docker Desktop
                subprocess.run(['open', '/Applications/Docker.app'], check=False)
                return True
            elif sys.platform.startswith("linux"):
                # Try systemctl for Linux
                subprocess.run(['sudo', 'systemctl', 'start', 'docker'], check=False)
                return True
            elif sys.platform == "win32":
                # Try Docker Desktop for Windows
                subprocess.run(['start', '', 'Docker Desktop'], shell=True, check=False)
                return True
        except:
            pass
        return False
    
    @staticmethod
    def get_docker_status():
        """Get comprehensive Docker status"""
        status = {
            'installed': DockerManager.is_docker_installed(),
            'running': False,
            'containers': [],
            'images': []
        }
        
        if status['installed']:
            status['running'] = DockerManager.is_docker_running()
            
            if status['running']:
                # Get containers
                try:
                    result = subprocess.run(['docker', 'ps', '-a', '--format', 'json'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                status['containers'].append(json.loads(line))
                except:
                    pass
                
                # Get images
                try:
                    result = subprocess.run(['docker', 'images', '--format', 'json'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                status['images'].append(json.loads(line))
                except:
                    pass
        
        return status

class ELIEProcessor:
    """Handles ELIE pipeline processing"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent  # Go up one level from interfaces/
        self.log_queue = queue.Queue()
        self.current_process = None
    
    
    def update_progress_from_output(self, output_line):
        """Update progress based on pipeline output"""
        output_lower = output_line.lower()
        
        # Track progress from pipeline output
        if 'step' in output_lower and '===' in output_lower:
            st.session_state.logs.append(output_line)
        elif 'completed successfully' in output_lower:
            st.session_state.logs.append(output_line)
        
        # Get pipeline type from session state if available
        pipeline_type = getattr(st.session_state, 'current_pipeline_type', 'MLI')
        
        # ── Gemini pipeline: 7 steps ──
        # Step 1: Detection | Step 2: Classification+Rotation | Step 3: OCR
        # Step 4: Post-processing | Step 5: Entity Recognition
        # Step 6: Crop Labels | Step 7: Cleanup
        #
        # ── Traditional MLI/SLI pipelines: 6 steps (unchanged) ──
        if pipeline_type == 'Gemini':
            if 'step 1:' in output_lower:
                st.session_state.pipeline_progress = 10
                st.session_state.current_stage = "🔍 Detection"
            elif 'step 2:' in output_lower:
                st.session_state.pipeline_progress = 25
                st.session_state.current_stage = "🏷️ Classification + Rotation"
            elif 'step 3:' in output_lower:
                st.session_state.pipeline_progress = 40
                st.session_state.current_stage = "📖 OCR"
            elif 'step 4:' in output_lower:
                st.session_state.pipeline_progress = 60
                st.session_state.current_stage = "🔧 Post-processing"
            elif 'step 5:' in output_lower:
                st.session_state.pipeline_progress = 75
                st.session_state.current_stage = "🧬 Entity Recognition"
            elif 'step 6:' in output_lower:
                st.session_state.pipeline_progress = 88
                st.session_state.current_stage = "✂️ Crop Labels"
            elif 'step 7:' in output_lower:
                st.session_state.pipeline_progress = 95
                st.session_state.current_stage = "🧹 Cleanup"
        else:
            # Traditional MLI / SLI pipelines
            if 'step 1:' in output_lower:
                if pipeline_type == 'MLI' and 'detection' in output_lower:
                    st.session_state.pipeline_progress = 15
                    st.session_state.current_stage = "🔍 Detection"
                else:
                    st.session_state.pipeline_progress = 20
                    st.session_state.current_stage = "🚫 Empty/Not-Empty Classification"
            elif 'step 2:' in output_lower:
                st.session_state.pipeline_progress = 35
                st.session_state.current_stage = "🏷️ ID/Description Classification"
            elif 'step 3:' in output_lower:
                st.session_state.pipeline_progress = 50
                st.session_state.current_stage = "✍️ Handwritten/Printed Classification"
            elif 'step 4:' in output_lower:
                st.session_state.pipeline_progress = 65
                st.session_state.current_stage = "🔄 Rotation Correction"
            elif 'step 5:' in output_lower:
                st.session_state.pipeline_progress = 80
                st.session_state.current_stage = "📖 OCR Processing"
            elif 'step 6:' in output_lower:
                st.session_state.pipeline_progress = 92
                st.session_state.current_stage = "🔧 Post-processing"
        
        # Track label / image counts from pipeline output
        if 'generated' in output_lower and 'crops' in output_lower:
            # e.g. "0a0b11...jpg generated 8 crops"
            import re as _re
            m = _re.search(r'generated\s+(\d+)\s+crops', output_lower)
            if m:
                st.session_state.setdefault('labels_detected', 0)
                st.session_state.labels_detected += int(m.group(1))
        elif 'processed' in output_lower and output_lower.strip().startswith('processed'):
            # Entity recognition: "Processed label_x in 1.2s"
            st.session_state.setdefault('labels_processed', 0)
            st.session_state.labels_processed += 1
        elif 'found' in output_lower and 'images to process' in output_lower:
            import re as _re
            m = _re.search(r'found\s+(\d+)\s+images', output_lower)
            if m:
                st.session_state.images_to_process = int(m.group(1))
        
        if 'pipeline completed successfully' in output_lower or '✅ pipeline completed successfully' in output_lower:
            st.session_state.pipeline_progress = 100
            st.session_state.current_stage = "✅ Completed"
            st.session_state.processing = False
        
    def get_input_images(self, input_dir):
        """Get list of input images"""
        input_path = Path(input_dir)
        if not input_path.exists():
            return []
        
        # If the user pointed at a single file, return it directly
        if input_path.is_file():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            return [input_path] if input_path.suffix.lower() in image_extensions else []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return [f for f in input_path.iterdir() 
                if f.suffix.lower() in image_extensions]
    
    def run_pipeline(self, pipeline_type, input_dir, output_dir, progress_callback=None):
        """Run the selected pipeline"""
        try:
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Select pipeline script
            if pipeline_type == "MLI":
                script_path = self.project_root / "tools" / "pipelines" / "run_mli_pipeline_conda.sh"
            elif pipeline_type == "SLI":
                script_path = self.project_root / "tools" / "pipelines" / "run_sli_pipeline_conda.sh"
            else:  # Gemini
                script_path = self.project_root / "tools" / "pipelines" / "run_gemini_pipeline_conda.sh"
            
            if not script_path.exists():
                raise FileNotFoundError(f"Pipeline script not found: {script_path}")
            
            # Start process
            self.current_process = subprocess.Popen(
                [str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.project_root,
                env=dict(os.environ, 
                        INPUT_DIR=str(input_dir),
                        OUTPUT_DIR=str(output_dir))
            )
            
            # Stream output
            for line in self.current_process.stdout:
                self.log_queue.put(line.strip())
                if progress_callback:
                    progress_callback(line.strip())
            
            self.current_process.wait()
            return self.current_process.returncode == 0
            
        except Exception as e:
            self.log_queue.put(f"Error: {str(e)}")
            return False
    
    def stop_pipeline(self):
        """Stop the current pipeline"""
        if self.current_process:
            self.current_process.terminate()
            self.current_process = None

def display_docker_status():
    """Display Docker status in sidebar"""
    st.sidebar.header("🐳 Docker Status")
    
    status = DockerManager.get_docker_status()
    
    # Installation status
    if status['installed']:
        st.sidebar.success("✅ Docker installed")
    else:
        st.sidebar.error("❌ Docker not installed")
        st.sidebar.markdown("[Install Docker](https://docs.docker.com/get-docker/)")
        return False
    
    # Running status
    if status['running']:
        st.sidebar.success("✅ Docker running")
    else:
        st.sidebar.warning("⚠️ Docker not running")
        if st.sidebar.button("🚀 Start Docker"):
            with st.sidebar:
                with st.spinner("Starting Docker..."):
                    if DockerManager.start_docker():
                        st.success("Docker start initiated")
                        time.sleep(3)
                        st.rerun()
                    else:
                        st.error("Failed to start Docker")
        return False
    
    # Container and image info
    if status['containers']:
        st.sidebar.write(f"📦 {len(status['containers'])} containers")
    if status['images']:
        st.sidebar.write(f"🖼️ {len(status['images'])} images")
    
    return True

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown(
        "<h1 style='margin-bottom:0'>🔬 ELIE</h1>"
        "<p style='color:#1976D2; margin-top:0; font-size:1.1rem'>Entomological Label Information Extraction</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Initialize all session state variables first
    if 'processor' not in st.session_state:
        st.session_state.processor = ELIEProcessor()
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        
    if 'logs' not in st.session_state:
        st.session_state.logs = []
        
    if 'pipeline_stages' not in st.session_state:
        st.session_state.pipeline_stages = []
        
    if 'metrics_data' not in st.session_state:
        st.session_state.metrics_data = []
        
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
        
    if 'job_duration' not in st.session_state:
        st.session_state.job_duration = None
    
    # Check Docker status
    docker_ready = display_docker_status()
    
    # Main interface
    if not docker_ready:
        st.error("🚫 Docker is required to run ELIE pipelines. Please start Docker to continue.")
        st.info("💡 Docker will start automatically when detected. Please wait a moment after clicking 'Start Docker'.")
        return
    
    # Pipeline configuration
    st.header("⚙️ Pipeline Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pipeline_type = st.selectbox(
            "Select Pipeline Type",
            ["MLI", "SLI", "Gemini"],
            help=(
                "MLI: Multi-Label (full specimen photos, traditional classifiers)\n"
                "SLI: Single-Label (pre-cropped labels, traditional classifiers)\n"
                "Gemini: Uses Gemini API for classification, rotation, and optionally OCR/HTR"
            )
        )
        
        # For Gemini pipeline, select pipeline mode and OCR engine
        if pipeline_type == "Gemini":
            gemini_mode = st.selectbox(
                "Image Type",
                ["MLI", "SLI"],
                help="MLI: Full specimen photos (runs detection first). SLI: Pre-cropped labels."
            )
            ocr_engine = st.selectbox(
                "OCR Engine",
                ["Gemini", "Tesseract", "Google Vision"],
                help=(
                    "Gemini: Handles printed, handwritten, and mixed labels\n"
                    "Tesseract: Printed labels only (local, free)\n"
                    "Google Vision: Printed labels only (requires credentials)"
                )
            )
            st.markdown("---")
            st.markdown("**Post-OCR Options**")
            enable_entity_recognition = st.checkbox(
                "Entity Recognition",
                value=False,
                help="Extract structured biodiversity entities (scientific names, collectors, dates, geography) from OCR text using Gemini, with GBIF validation and OSM geocoding."
            )
            if enable_entity_recognition:
                er_col1, er_col2, er_col3 = st.columns(3)
                with er_col1:
                    export_dwc = st.checkbox("Darwin Core JSON", value=True, help="Export records in Darwin Core format")
                with er_col2:
                    export_opends = st.checkbox("OpenDS JSON", value=False, help="Export records in OpenDS format")
                with er_col3:
                    export_csv = st.checkbox("DwC CSV", value=False, help="Export Darwin Core records as CSV")
            else:
                export_dwc = False
                export_opends = False
                export_csv = False
            enable_crop_labels = st.checkbox(
                "Crop Labels",
                value=False,
                help="Crop individual label regions from original images using detected bounding boxes."
            )
        else:
            gemini_mode = None
            ocr_engine = None
            enable_entity_recognition = False
            export_dwc = False
            export_opends = False
            export_csv = False
            enable_crop_labels = False
        
        # Input directory
        if pipeline_type == "Gemini" and gemini_mode:
            default_input = str(Path(__file__).parent.parent / "data" / gemini_mode / "input")
        else:
            default_input = str(Path(__file__).parent.parent / "data" / pipeline_type / "input")
        input_dir = st.text_input(
            "Input Directory", 
            value=default_input,
            help="Directory containing images to process"
        )
    
    with col2:
        # Output directory
        if pipeline_type == "Gemini" and gemini_mode:
            default_output = str(Path(__file__).parent.parent / "data" / gemini_mode / "output")
        else:
            default_output = str(Path(__file__).parent.parent / "data" / pipeline_type / "output")
        output_dir = st.text_input(
            "Output Directory",
            value=default_output,
            help="Directory where results will be saved"
        )
        
        # Processing options
        batch_size = st.slider("Batch Size", 1, 8, 1, help="Number of images to process simultaneously")
    
    # API credentials section
    if pipeline_type == "Gemini" or (ocr_engine and ocr_engine == "Google Vision"):
        st.subheader("🔑 API Credentials")
        cred_col1, cred_col2 = st.columns(2)
        
        with cred_col1:
            if pipeline_type == "Gemini":
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Your Gemini API key (from Google AI Studio). Required for Gemini pipeline.",
                    placeholder="Enter your Gemini API key"
                )
            else:
                gemini_api_key = None
        
        with cred_col2:
            if ocr_engine == "Google Vision":
                vision_credentials = st.text_input(
                    "Google Vision Credentials Path",
                    help="Path to your Google Cloud Vision API credentials JSON file.",
                    placeholder="/path/to/credentials.json"
                )
            else:
                vision_credentials = None
    else:
        gemini_api_key = None
        vision_credentials = None
    
    # Input validation and preview
    st.subheader("📁 Input Data")
    
    input_path = Path(input_dir)
    if input_path.exists():
        images = st.session_state.processor.get_input_images(input_dir)
        
        if images:
            st.success(f"Found {len(images)} images in input directory")
            
            # Show sample images
            if st.checkbox("Preview Input Images"):
                cols = st.columns(min(4, len(images)))
                for i, img_path in enumerate(images[:4]):
                    with cols[i]:
                        try:
                            img = Image.open(img_path)
                            # Cap preview to 400px wide
                            max_w = 400
                            if img.width > max_w:
                                ratio = max_w / img.width
                                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
                            st.image(img, caption=img_path.name, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading {img_path.name}")
        else:
            st.warning("No images found in input directory")
            st.info("Supported formats: JPG, PNG, BMP, TIFF")
    else:
        st.error("Input directory does not exist")
        if st.button("Create Input Directory"):
            input_path.mkdir(parents=True, exist_ok=True)
            st.success(f"Created directory: {input_path}")
            st.rerun()
    
    # Processing controls
    st.header("🚀 Processing Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Processing", 
                    disabled=st.session_state.processing or not images if 'images' in locals() else True):
            # Clear output directory for fresh results
            output_path = Path(output_dir)
            if output_path.exists():
                import shutil
                try:
                    shutil.rmtree(output_path)
                    output_path.mkdir(parents=True, exist_ok=True)
                    st.success(f"🗑️ Cleared output directory: {output_dir}")
                except Exception as e:
                    st.warning(f"Could not clear output directory: {e}")
            else:
                output_path.mkdir(parents=True, exist_ok=True)
            
            # Reset processing state
            st.session_state.processing = True
            st.session_state.logs = []
            st.session_state.processing_step = 0
            st.session_state.start_time = datetime.now()
            st.session_state.last_log_position = 0
            st.session_state.metrics_data = []  # Clear metrics too
            st.session_state.job_duration = None  # Reset duration
            st.session_state.current_pipeline_type = pipeline_type  # Set pipeline type for progress tracking
            # Reset progress tracking
            st.session_state.pipeline_progress = 0
            st.session_state.current_stage = "Starting..."
            st.session_state.labels_detected = 0
            st.session_state.labels_processed = 0
            st.session_state.images_to_process = 0
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Processing", disabled=not st.session_state.processing):
            # Stop the real pipeline process
            current_time = datetime.now().strftime('%H:%M:%S')
            
            if st.session_state.pipeline_process is not None:
                try:
                    st.session_state.pipeline_process.terminate()
                    st.session_state.logs.append(f"[{current_time}] ⏹️ Pipeline terminated by user")
                except:
                    st.session_state.logs.append(f"[{current_time}] ⏹️ Failed to terminate pipeline")
                
                st.session_state.pipeline_process = None
                
                # Clean up temp file
                if st.session_state.pipeline_output_file and os.path.exists(st.session_state.pipeline_output_file):
                    try:
                        os.unlink(st.session_state.pipeline_output_file)
                    except:
                        pass
                    st.session_state.pipeline_output_file = None
            
            # Reset processing state
            st.session_state.processing = False
            st.rerun()
    
    with col3:
        if st.button("📁 Open Output Folder"):
            if Path(output_dir).exists():
                if sys.platform == "darwin":
                    subprocess.run(["open", output_dir])
                elif sys.platform.startswith("linux"):
                    subprocess.run(["xdg-open", output_dir])
                elif sys.platform == "win32":
                    subprocess.run(["explorer", output_dir])
            else:
                st.warning("Output directory does not exist yet")
    
    # Processing status and logs
    if st.session_state.processing:
        st.header("📊 Real-time Processing Dashboard")
        
        # Initialize simple progress tracking
        if 'pipeline_progress' not in st.session_state:
            st.session_state.pipeline_progress = 0
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = "Starting..."
        
        # Initialize processing start time
        if not hasattr(st.session_state, 'start_time') or st.session_state.start_time is None:
            st.session_state.start_time = datetime.now()
        # Overall progress section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Use simple progress tracking
            overall_progress = st.session_state.pipeline_progress
            
            st.metric("Overall Progress", f"{overall_progress}%", st.session_state.current_stage)
            st.progress(overall_progress / 100)
        
        with col2:
            # Processing time
            if hasattr(st.session_state, 'start_time') and st.session_state.start_time is not None:
                elapsed_time = datetime.now() - st.session_state.start_time
                hours = elapsed_time.seconds // 3600
                minutes = (elapsed_time.seconds % 3600) // 60
                seconds = elapsed_time.seconds % 60
                if hours > 0:
                    time_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"
                st.metric("Processing Time", time_str)
            else:
                st.metric("Processing Time", "0s")
        
        with col3:
            # Labels detected / processed
            labels_det = getattr(st.session_state, 'labels_detected', 0)
            labels_proc = getattr(st.session_state, 'labels_processed', 0)
            if labels_det > 0:
                st.metric("Labels Detected", labels_det)
                if labels_proc > 0:
                    st.metric("Labels Processed", f"{labels_proc}/{labels_det}")
            elif 'images' in locals():
                st.metric("Input Images", len(images))
        
        
        # Real-time metrics chart
        st.subheader("📈 Processing Metrics")
        
        # Add new data point safely
        current_time = datetime.now()
        new_data_point = {
            'timestamp': current_time,
            'progress': overall_progress,
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(interval=None)
        }
        
        # Update metrics data
        st.session_state.metrics_data.append(new_data_point)
        
        # Keep only last 30 data points for performance
        if len(st.session_state.metrics_data) > 30:
            st.session_state.metrics_data = st.session_state.metrics_data[-30:]
        
        # Show basic metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current CPU", f"{new_data_point['cpu_usage']:.1f}%")
            
        with col2:
            st.metric("Current Memory", f"{new_data_point['memory_usage']:.1f}%")
        
        # Create charts only if we have enough data
        if len(st.session_state.metrics_data) > 2:
            try:
                df = pd.DataFrame(st.session_state.metrics_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Progress over time
                    fig_progress = px.line(df, x='timestamp', y='progress',
                                         title='Processing Progress Over Time',
                                         labels={'progress': 'Progress (%)', 'timestamp': 'Time'})
                    fig_progress.update_layout(height=250, showlegend=False)
                    fig_progress.update_traces(line_color='#1f77b4', line_width=3)
                    st.plotly_chart(fig_progress, config={'displayModeBar': False})
                
                with col2:
                    # System resources
                    fig_resources = go.Figure()
                    fig_resources.add_trace(go.Scatter(x=df['timestamp'], y=df['memory_usage'],
                                                     mode='lines', name='Memory %', line=dict(color='orange')))
                    fig_resources.add_trace(go.Scatter(x=df['timestamp'], y=df['cpu_usage'],
                                                     mode='lines', name='CPU %', line=dict(color='red')))
                    fig_resources.update_layout(title='System Resources', height=250,
                                              yaxis_title='Usage (%)', xaxis_title='Time')
                    st.plotly_chart(fig_resources, config={'displayModeBar': False})
            except Exception as e:
                st.info("Charts will appear once processing starts...")
        
        # Enhanced log viewer
        st.subheader("📝 Processing Logs")
        
        # Real pipeline execution using subprocess
        # Initialize pipeline process tracking
        if 'pipeline_process' not in st.session_state:
            st.session_state.pipeline_process = None
        if 'pipeline_output_file' not in st.session_state:
            st.session_state.pipeline_output_file = None
        if 'last_log_position' not in st.session_state:
            st.session_state.last_log_position = 0
            
        # Start real pipeline if processing
        if st.session_state.processing:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Start the pipeline if not already started
            if st.session_state.pipeline_process is None:
                # Determine which pipeline script to run
                if pipeline_type == "MLI":
                    script_path = st.session_state.processor.project_root / "tools" / "pipelines" / "run_mli_pipeline_conda.sh"
                elif pipeline_type == "SLI":
                    script_path = st.session_state.processor.project_root / "tools" / "pipelines" / "run_sli_pipeline_conda.sh"
                else:  # Gemini
                    script_path = st.session_state.processor.project_root / "tools" / "pipelines" / "run_gemini_pipeline_conda.sh"
                
                # Validate input directory first
                input_path = Path(input_dir)
                if not input_path.exists():
                    st.session_state.logs.append(f"[{current_time}] ❌ Input directory does not exist: {input_dir}")
                    st.session_state.processing = False
                    return
                
                # Check for images in input directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                if input_path.is_dir():
                    input_images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
                elif input_path.is_file() and input_path.suffix.lower() in image_extensions:
                    input_images = [input_path]
                else:
                    input_images = []
                if not input_images:
                    st.session_state.logs.append(f"[{current_time}] ⚠️ No images found in input directory: {input_dir}")
                    st.session_state.logs.append(f"[{current_time}] 🗂️ Supported formats: {', '.join(image_extensions)}")
                else:
                    st.session_state.logs.append(f"[{current_time}] 🗂️ Found {len(input_images)} images in input directory")
                
                # Check if script exists
                if script_path.exists():
                    try:
                        # Set up environment variables for paths
                        env = os.environ.copy()
                        env['INPUT_DIR'] = input_dir
                        env['OUTPUT_DIR'] = output_dir
                        
                        # Set Gemini pipeline environment variables
                        if pipeline_type == "Gemini":
                            if gemini_api_key:
                                env['GEMINI_API_KEY'] = gemini_api_key
                            if gemini_mode:
                                env['PIPELINE_MODE'] = gemini_mode
                            if ocr_engine:
                                engine_map = {"Gemini": "gemini", "Tesseract": "tesseract", "Google Vision": "vision"}
                                env['OCR_ENGINE'] = engine_map.get(ocr_engine, "gemini")
                            if vision_credentials:
                                env['GOOGLE_VISION_CREDENTIALS'] = vision_credentials
                            # Entity recognition flags
                            if enable_entity_recognition:
                                env['ENTITY_RECOGNITION'] = 'true'
                                if export_dwc:
                                    env['EXPORT_DWC'] = 'true'
                                if export_opends:
                                    env['EXPORT_OPENDS'] = 'true'
                                if export_csv:
                                    env['EXPORT_CSV'] = 'true'
                            # Crop labels flag
                            if enable_crop_labels:
                                env['CROP_LABELS'] = 'true'
                        
                        # Create a temporary file to capture output
                        output_fd, st.session_state.pipeline_output_file = tempfile.mkstemp(suffix='.log')
                        os.close(output_fd)
                        
                        # Start the pipeline process
                        st.session_state.logs.append(f"[{current_time}] 🚀 Starting {pipeline_type} pipeline...")
                        st.session_state.logs.append(f"[{current_time}] 📂 Input: {input_dir}")
                        st.session_state.logs.append(f"[{current_time}] 📁 Output: {output_dir}")
                        st.session_state.logs.append(f"[{current_time}] 🖥️ Script: {script_path}")
                        st.session_state.logs.append(f"[{current_time}] 🚀 Environment: INPUT_DIR={input_dir}, OUTPUT_DIR={output_dir}")
                        
                        # Initialize pipeline progress
                        st.session_state.pipeline_progress = 5
                        st.session_state.current_stage = "🚀 Starting pipeline..."
                        
                        st.session_state.pipeline_process = subprocess.Popen(
                            [str(script_path)],
                            stdout=open(st.session_state.pipeline_output_file, 'w', buffering=1),
                            stderr=subprocess.STDOUT,
                            cwd=st.session_state.processor.project_root,
                            env=env,
                            universal_newlines=True
                        )
                        
                    except Exception as e:
                        st.session_state.logs.append(f"[{current_time}] ❌ Error starting pipeline: {str(e)}")
                        st.session_state.processing = False
                        st.session_state.pipeline_process = None
                else:
                    st.session_state.logs.append(f"[{current_time}] ❌ Pipeline script not found: {script_path}")
                    st.session_state.processing = False
            
            # Check pipeline progress
            if st.session_state.pipeline_process is not None:
                # Check if process is still running
                poll = st.session_state.pipeline_process.poll()
                
                # Track process status changes
                if 'last_poll_status' not in st.session_state:
                    st.session_state.last_poll_status = None
                if poll != st.session_state.last_poll_status:
                    st.session_state.last_poll_status = poll
                
                if poll is None:
                    # Process is still running - read latest output
                    if st.session_state.pipeline_output_file and os.path.exists(st.session_state.pipeline_output_file):
                        try:
                            with open(st.session_state.pipeline_output_file, 'r') as f:
                                # Seek to last position we read
                                f.seek(st.session_state.last_log_position)
                                new_content = f.read()
                                
                                if new_content:
                                    # Update position
                                    st.session_state.last_log_position = f.tell()
                                    
                                    # Process new lines
                                    new_lines = new_content.strip().split('\n')
                                    for line in new_lines:
                                        clean_line = line.strip()
                                        if clean_line:
                                            st.session_state.logs.append(f"[{current_time}] {clean_line}")
                                            
                                            # Update progress based on output
                                            st.session_state.processor.update_progress_from_output(clean_line)
                                            
                                            # Check for completion in the logs as backup (be more specific)
                                            if ('pipeline completed successfully' in clean_line.lower() and '✅ pipeline completed successfully' in clean_line) or 'final output captured' in clean_line.lower():
                                                # Force completion
                                                st.session_state.pipeline_progress = 100
                                                st.session_state.current_stage = "✅ Completed"
                                                st.session_state.processing = False
                                                # Clean up process
                                                if st.session_state.pipeline_process:
                                                    st.session_state.pipeline_process = None
                                        
                        except Exception as e:
                            pass  # Continue without error if file reading fails
                            
                else:
                    # Process completed
                    st.session_state.processing = False
                    
                    # Read final output
                    if st.session_state.pipeline_output_file and os.path.exists(st.session_state.pipeline_output_file):
                        try:
                            with open(st.session_state.pipeline_output_file, 'r') as f:
                                final_output = f.read()
                                if final_output.strip():
                                    st.session_state.logs.append(f"[{current_time}] Final output captured")
                            
                            # Clean up temp file
                            os.unlink(st.session_state.pipeline_output_file)
                        except:
                            pass
                    
                    if poll == 0:
                        # Calculate total job duration
                        if hasattr(st.session_state, 'start_time') and st.session_state.start_time is not None:
                            total_duration = datetime.now() - st.session_state.start_time
                            hours = total_duration.seconds // 3600
                            minutes = (total_duration.seconds % 3600) // 60
                            seconds = total_duration.seconds % 60
                            if hours > 0:
                                duration_str = f"{hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                duration_str = f"{minutes}m {seconds}s"
                            else:
                                duration_str = f"{seconds}s"
                            st.session_state.job_duration = duration_str
                        else:
                            st.session_state.job_duration = "Unknown"
                        
                        st.session_state.logs.append(f"[{current_time}] ✅ Pipeline completed successfully! (Total time: {st.session_state.job_duration})")
                        st.session_state.pipeline_progress = 100
                        st.session_state.current_stage = "✅ Completed"
                        # Force immediate UI refresh to show completion
                        time.sleep(0.5)  # Brief pause to ensure state is updated
                        st.rerun()
                    else:
                        st.session_state.logs.append(f"[{current_time}] ❌ Pipeline failed with exit code {poll}")
                    
                    st.session_state.pipeline_process = None
                    st.session_state.pipeline_output_file = None
        
        # Enhanced log display with filtering
        # Initialize log display preferences in session state to avoid rerun issues
        if 'log_level' not in st.session_state:
            st.session_state.log_level = "All"
        if 'max_lines' not in st.session_state:
            st.session_state.max_lines = 30
        
        # Show log controls only when not processing to avoid rerun issues
        if not st.session_state.processing:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.session_state.log_level = st.selectbox(
                    "Log Level", 
                    ["All", "Info", "Warning", "Error"],
                    index=["All", "Info", "Warning", "Error"].index(st.session_state.log_level)
                )
                st.session_state.max_lines = st.slider(
                    "Max Lines", 
                    10, 100, 
                    st.session_state.max_lines
                )
        else:
            # During processing, just show a simple header
            st.write(f"**Live Processing Logs** (showing last {st.session_state.max_lines} lines)")
        
        # Display logs using stored preferences
        if st.session_state.logs:
            # Filter logs based on level
            filtered_logs = st.session_state.logs
            if st.session_state.log_level != "All":
                filtered_logs = [log for log in st.session_state.logs 
                               if st.session_state.log_level.lower() in log.lower()]
            
            # Display logs with syntax highlighting
            log_text = "\n".join(filtered_logs[-st.session_state.max_lines:])
            st.code(log_text, language="bash")
        
        # Auto-refresh during processing
        if st.session_state.processing:
            # Refresh every 2 seconds during real pipeline execution
            time.sleep(2.0)
            st.rerun()
        else:
            # Keep showing the results without constant refresh
            pass
    
    # Results section - only show if not currently processing or if processing is complete
    if not st.session_state.processing or (hasattr(st.session_state, 'pipeline_progress') and st.session_state.pipeline_progress == 100):
        st.header("📈 Results & Analysis")
        
        # Show job completion info if available
        if hasattr(st.session_state, 'job_duration') and st.session_state.job_duration:
            st.success(f"✅ Job completed in {st.session_state.job_duration}")
        
        output_path = Path(output_dir)
        input_path = Path(input_dir)
        
        if output_path.exists():
            # Look for result files
            result_files = list(output_path.glob("*.json")) + list(output_path.glob("*.csv"))
            
            # Load consolidated results (single source of truth)
            consolidated = _load_consolidated(output_path)
            bbox_data = _group_labels_by_image(consolidated)
            
            # Initialise OCR edits store in session state
            if 'ocr_edits' not in st.session_state:
                st.session_state.ocr_edits = {}
            if 'entity_edits' not in st.session_state:
                st.session_state.entity_edits = {}
            if 'validated' not in st.session_state:
                st.session_state.validated = False
            
            if consolidated or result_files:
                # ─── Interactive Label Explorer ───
                st.subheader("🔍 Label Explorer")
                st.caption("Browse images, review and correct OCR results, then validate.")
                
                # Collect input images (all formats)
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                if input_path.exists() and input_path.is_dir():
                    input_images = sorted(
                        f for f in input_path.iterdir()
                        if f.suffix.lower() in image_extensions and not f.name.startswith("._")
                    )
                elif input_path.exists() and input_path.is_file() and input_path.suffix.lower() in image_extensions:
                    input_images = [input_path]
                else:
                    input_images = []
                
                # Color maps
                color_map = {
                    "printed": (0, 180, 0), "handwritten": (0, 120, 255),
                    "mixed": (255, 165, 0), "identifier": (255, 0, 0),
                    "empty": (128, 128, 128),
                }
                hex_color_map = {
                    "printed": "#00B400", "handwritten": "#0078FF",
                    "mixed": "#FFA500", "identifier": "#FF0000",
                    "empty": "#808080",
                }
                
                if input_images:
                    selected_name = st.selectbox(
                        "Select an image",
                        [img.name for img in input_images],
                        key="explorer_image_select"
                    )
                    selected_path = input_path / selected_name
                    labels_for_image = bbox_data.get(selected_name, [])
                    
                    try:
                        img = Image.open(selected_path)
                        img_cv = cv2.imread(str(selected_path))
                        
                        # --- Side-by-side: image (left) + editable labels (right) ---
                        img_col, info_col = st.columns([1, 1])
                        
                        with img_col:
                            MAX_DISPLAY_W = 800  # cap image sent to browser
                            if labels_for_image and img_cv is not None:
                                annotated = img_cv.copy()
                                h, w = annotated.shape[:2]
                                thickness = max(1, int(min(h, w) / 400))
                                font_scale = max(0.3, min(h, w) / 800)
                                for lbl in labels_for_image:
                                    bbox = lbl.get("bbox", {})
                                    x1 = int(bbox.get("xmin", lbl.get("xmin", 0)))
                                    y1 = int(bbox.get("ymin", lbl.get("ymin", 0)))
                                    x2 = int(bbox.get("xmax", lbl.get("xmax", 0)))
                                    y2 = int(bbox.get("ymax", lbl.get("ymax", 0)))
                                    cat = lbl.get("category", "label")
                                    color = color_map.get(cat, (0, 255, 0))
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                                    txt = f"{lbl.get('label_index', '')} {cat}"
                                    cv2.putText(annotated, txt, (x1, max(y1 - 4, 12)),
                                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                                # Downscale for browser
                                if w > MAX_DISPLAY_W:
                                    scale = MAX_DISPLAY_W / w
                                    annotated = cv2.resize(annotated, (MAX_DISPLAY_W, int(h * scale)))
                                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                st.image(annotated_rgb,
                                         caption=f"{selected_name} — {len(labels_for_image)} labels",
                                         use_container_width=True)
                            else:
                                # Downscale plain image too
                                display_img = img.copy()
                                if display_img.width > MAX_DISPLAY_W:
                                    ratio = MAX_DISPLAY_W / display_img.width
                                    display_img = display_img.resize(
                                        (MAX_DISPLAY_W, int(display_img.height * ratio)), Image.LANCZOS
                                    )
                                st.image(display_img, caption=selected_name, use_container_width=True)
                        
                        with info_col:
                            if labels_for_image:
                                for lbl_idx, lbl in enumerate(labels_for_image):
                                    idx = lbl.get("label_index", "?")
                                    cat = lbl.get("category", "unknown")
                                    det_conf = lbl.get("detection_confidence", lbl.get("confidence", ""))
                                    angle = lbl.get("rotation_angle", "-")
                                    cat_color = hex_color_map.get(cat, "#00FF00")
                                    label_fn = lbl.get("label_filename", f"{selected_name}_{idx}")
                                    
                                    # Get OCR text: user edit > consolidated ocr.text
                                    ocr_block = lbl.get("ocr", {})
                                    original_text = ocr_block.get("text", "") if isinstance(ocr_block, dict) else ""
                                    current_text = st.session_state.ocr_edits.get(label_fn, original_text)
                                    
                                    # Header with label info
                                    st.markdown(
                                        f"**Label {idx}** — "
                                        f"<span style='color:{cat_color}; font-weight:bold'>{cat}</span>"
                                        f" &nbsp; (conf: {det_conf}, rot: {angle}°)",
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Try to extract and display the label crop if available
                                    crop_path = None
                                    if img_cv is not None:
                                        bbox = lbl.get("bbox", {})
                                        x1 = int(bbox.get("xmin", lbl.get("xmin", 0)))
                                        y1 = int(bbox.get("ymin", lbl.get("ymin", 0)))
                                        x2 = int(bbox.get("xmax", lbl.get("xmax", img_cv.shape[1])))
                                        y2 = int(bbox.get("ymax", lbl.get("ymax", img_cv.shape[0])))
                                        
                                        # Validate bbox coordinates
                                        x1, y1 = max(0, x1), max(0, y1)
                                        x2 = min(img_cv.shape[1], max(x1 + 10, x2))
                                        y2 = min(img_cv.shape[0], max(y1 + 10, y2))
                                        
                                        if x2 > x1 and y2 > y1:
                                            # Extract label crop
                                            label_crop = img_cv[y1:y2, x1:x2]
                                            # Resize for display if too large
                                            crop_h, crop_w = label_crop.shape[:2]
                                            max_crop_h = 200
                                            if crop_h > max_crop_h:
                                                scale = max_crop_h / crop_h
                                                label_crop = cv2.resize(label_crop, 
                                                                       (int(crop_w * scale), max_crop_h))
                                            label_crop_rgb = cv2.cvtColor(label_crop, cv2.COLOR_BGR2RGB)
                                            
                                            # Display crop and OCR side-by-side
                                            crop_col, text_col = st.columns([0.4, 0.6])
                                            
                                            with crop_col:
                                                st.image(label_crop_rgb, caption="Label Image", use_container_width=True)
                                            
                                            with text_col:
                                                # Editable OCR text
                                                edited = st.text_area(
                                                    f"OCR Text",
                                                    value=current_text,
                                                    key=f"ocr_{label_fn}",
                                                    height=150,
                                                    label_visibility="collapsed",
                                                    placeholder="Edit OCR text here..."
                                                )
                                                # Track edits
                                                if edited != original_text:
                                                    st.session_state.ocr_edits[label_fn] = edited
                                                    st.caption("✏️ Modified", unsafe_allow_html=False)
                                                elif label_fn in st.session_state.ocr_edits:
                                                    # User reverted to original
                                                    del st.session_state.ocr_edits[label_fn]
                                    
                                    if crop_path is None and img_cv is None:
                                        # Fallback: just show text area if image can't be processed
                                        edited = st.text_area(
                                            f"OCR Text for Label {idx}",
                                            value=current_text,
                                            key=f"ocr_{label_fn}",
                                            height=120,
                                            placeholder="Edit OCR text here..."
                                        )
                                        if edited != original_text:
                                            st.session_state.ocr_edits[label_fn] = edited
                                        elif label_fn in st.session_state.ocr_edits:
                                            del st.session_state.ocr_edits[label_fn]
                                    
                                    # --- Entity extraction data (if available) ---
                                    entities = lbl.get("entity_extraction", {})
                                    if entities:
                                        with st.expander("🧬 Extracted Entities", expanded=False):
                                            sci = entities.get("scientific_names", [])
                                            if sci and sci[0].get("name"):
                                                name_str = sci[0]["name"]
                                                auth = sci[0].get("authority", "")
                                                gbif = sci[0].get("gbif_validation", {})
                                                badge = "✅ GBIF" if gbif else "⚠️ unverified"
                                                st.markdown(f"**Scientific name:** *{name_str}* {auth} &nbsp; {badge}")
                                            if entities.get("recordedBy"):
                                                st.markdown(f"**Collector:** {entities['recordedBy']}")
                                            if entities.get("eventDate") or entities.get("verbatimEventDate"):
                                                st.markdown(f"**Date:** {entities.get('eventDate', entities.get('verbatimEventDate', ''))}")
                                            geo = entities.get("geographic_data", {})
                                            if geo.get("locality") or geo.get("verbatimLocality"):
                                                loc = geo.get("locality", geo.get("verbatimLocality", ""))
                                                country = geo.get("parsed", {}).get("country", "")
                                                loc_str = f"{loc}, {country}" if country else loc
                                                st.markdown(f"**Locality:** {loc_str}")
                                            traits = entities.get("traits_and_status", {})
                                            if traits.get("type_status"):
                                                st.markdown(f"**Type status:** {traits['type_status']}")
                                            if entities.get("institutionCode"):
                                                st.markdown(f"**Institution:** {entities['institutionCode']}")
                                    st.markdown("---")
                            else:
                                st.info("No label data for this image.")
                    except Exception as e:
                        st.error(f"Error loading {selected_name}: {e}")
                else:
                    st.info("No input images found.")
                
                # ─── Validation ───
                st.subheader("✅ Validation")
                edit_count = len(st.session_state.ocr_edits)
                total_labels = len(consolidated)
                
                if edit_count:
                    st.success(f"✏️ **{edit_count}** label(s) have been manually corrected.")
                    
                    # Show corrections summary
                    if st.checkbox("📋 Show corrections summary"):
                        correction_data = []
                        for entry in consolidated:
                            label_fn = entry.get("label_filename", "")
                            if label_fn in st.session_state.ocr_edits:
                                original = (entry.get("ocr", {}) or {}).get("text", "") if isinstance(entry.get("ocr"), dict) else ""
                                corrected = st.session_state.ocr_edits[label_fn]
                                correction_data.append({
                                    "Label": label_fn,
                                    "Original": original[:100] + "..." if len(original) > 100 else original,
                                    "Corrected": corrected[:100] + "..." if len(corrected) > 100 else corrected,
                                })
                        
                        df_corrections = pd.DataFrame(correction_data)
                        st.dataframe(df_corrections, use_container_width=True)
                else:
                    st.caption("✅ No corrections made — OCR text is unchanged.")
                
                val_col1, val_col2, val_col3 = st.columns(3)
                with val_col1:
                    if st.button("✅ Validate & Save", type="primary",
                                 help="Save current OCR text (with your corrections) as the validated results."):
                        # Build validated version of consolidated results
                        validated = []
                        for entry in consolidated:
                            v = dict(entry)  # shallow copy
                            label_fn = v.get("label_filename", "")
                            ocr = dict(v.get("ocr", {})) if isinstance(v.get("ocr"), dict) else {}
                            if label_fn in st.session_state.ocr_edits:
                                ocr["text"] = st.session_state.ocr_edits[label_fn]
                                ocr["manually_corrected"] = True
                            else:
                                ocr["manually_corrected"] = False
                            v["ocr"] = ocr
                            
                            # Merge entity edits if available
                            if label_fn in st.session_state.entity_edits:
                                entity_edits = st.session_state.entity_edits[label_fn]
                                if not v.get("entity_extraction"):
                                    v["entity_extraction"] = {}
                                # Update entity fields
                                if "scientific_name" in entity_edits:
                                    sci_names = v["entity_extraction"].get("scientific_names", [{}])
                                    sci_names[0]["name"] = entity_edits["scientific_name"]
                                if "authority" in entity_edits:
                                    sci_names = v["entity_extraction"].get("scientific_names", [{}])
                                    sci_names[0]["authority"] = entity_edits["authority"]
                                if "recordedBy" in entity_edits:
                                    v["entity_extraction"]["recordedBy"] = entity_edits["recordedBy"]
                                if "eventDate" in entity_edits:
                                    v["entity_extraction"]["eventDate"] = entity_edits["eventDate"]
                                if "type_status" in entity_edits:
                                    traits = v["entity_extraction"].get("traits_and_status", {})
                                    traits["type_status"] = entity_edits["type_status"]
                                    v["entity_extraction"]["traits_and_status"] = traits
                                if "institutionCode" in entity_edits:
                                    v["entity_extraction"]["institutionCode"] = entity_edits["institutionCode"]
                                if "locality" in entity_edits or "country" in entity_edits:
                                    geo = v["entity_extraction"].get("geographic_data", {})
                                    if "locality" in entity_edits:
                                        geo["locality"] = entity_edits["locality"]
                                    if "country" in entity_edits:
                                        parsed = geo.get("parsed", {})
                                        parsed["country"] = entity_edits["country"]
                                        geo["parsed"] = parsed
                                    v["entity_extraction"]["geographic_data"] = geo
                            
                            v["validated"] = True
                            validated.append(v)
                        # Save
                        validated_path = output_path / "validated_results.json"
                        with open(validated_path, "w", encoding="utf-8") as f:
                            json.dump(validated, f, indent=2, ensure_ascii=False)
                        st.session_state.validated = True
                        st.success(f"✅ Saved {len(validated)} labels → `validated_results.json` ({edit_count} corrected)")
                
                with val_col2:
                    if st.session_state.ocr_edits or st.session_state.entity_edits:
                        if st.button("↩️ Reset all corrections"):
                            st.session_state.ocr_edits = {}
                            st.session_state.entity_edits = {}
                            st.session_state.validated = False
                            st.rerun()
                
                with val_col3:
                    if edit_count > 0 or len(st.session_state.entity_edits) > 0:
                        # Create export data
                        export_data = []
                        for entry in consolidated:
                            label_fn = entry.get("label_filename", "")
                            row = {
                                "Label": label_fn,
                                "Original_OCR": (entry.get("ocr", {}) or {}).get("text", "") if isinstance(entry.get("ocr"), dict) else "",
                            }
                            if label_fn in st.session_state.ocr_edits:
                                row["Corrected_OCR"] = st.session_state.ocr_edits[label_fn]
                                row["OCR_Modified"] = "Yes"
                            else:
                                row["Corrected_OCR"] = row["Original_OCR"]
                                row["OCR_Modified"] = "No"
                            
                            # Add entity fields if edited
                            if label_fn in st.session_state.entity_edits:
                                edits = st.session_state.entity_edits[label_fn]
                                for key, value in edits.items():
                                    row[f"Entity_{key}"] = value
                            
                            export_data.append(row)
                        
                        df_export = pd.DataFrame(export_data)
                        csv = df_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Edits (CSV)",
                            data=csv,
                            file_name=f"label_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                
                # ─── Fine-grained Entity Field Editing ───
                st.subheader("🎯 Edit Entity Fields")
                st.caption("Correct specific structured fields (scientific name, collector, date, location, etc.) extracted from labels.")
                
                # Show entity editor if entity data exists
                if consolidated:
                    labels_with_entities = [lbl for lbl in consolidated if lbl.get("entity_extraction")]
                    
                    if labels_with_entities:
                        entity_label_names = [lbl.get("label_filename", f"Label {lbl.get('label_index', '?')}") 
                                            for lbl in labels_with_entities]
                        selected_entity_label = st.selectbox(
                            "Select label to edit entities",
                            entity_label_names,
                            key="entity_editor_select"
                        )
                        
                        selected_entity_idx = entity_label_names.index(selected_entity_label)
                        selected_entity_lbl = labels_with_entities[selected_entity_idx]
                        entities = selected_entity_lbl.get("entity_extraction", {})
                        label_fn = selected_entity_lbl.get("label_filename", "")
                        
                        # Initialize edits for this label if not exists
                        if label_fn not in st.session_state.entity_edits:
                            st.session_state.entity_edits[label_fn] = {}
                        
                        current_edits = st.session_state.entity_edits[label_fn]
                        
                        # Create columns for entity editing
                        entity_col1, entity_col2 = st.columns(2)
                        
                        with entity_col1:
                            st.markdown("**Taxonomy & Specimen**")
                            
                            # Scientific name
                            sci_names = entities.get("scientific_names", [])
                            original_sci_name = sci_names[0].get("name", "") if sci_names else ""
                            current_sci_name = current_edits.get("scientific_name", original_sci_name)
                            edited_sci_name = st.text_input(
                                "Scientific Name",
                                value=current_sci_name,
                                key=f"sci_name_{label_fn}",
                                help="Binomial nomenclature (genus species)"
                            )
                            if edited_sci_name != original_sci_name:
                                current_edits["scientific_name"] = edited_sci_name
                            elif "scientific_name" in current_edits:
                                del current_edits["scientific_name"]
                            
                            # Authority
                            sci_auth = sci_names[0].get("authority", "") if sci_names else ""
                            current_auth = current_edits.get("authority", sci_auth)
                            edited_auth = st.text_input(
                                "Authority",
                                value=current_auth,
                                key=f"auth_{label_fn}",
                                help="Author and year"
                            )
                            if edited_auth != sci_auth:
                                current_edits["authority"] = edited_auth
                            elif "authority" in current_edits:
                                del current_edits["authority"]
                            
                            # Type status
                            traits = entities.get("traits_and_status", {})
                            original_type_status = traits.get("type_status", "")
                            current_type_status = current_edits.get("type_status", original_type_status)
                            edited_type_status = st.text_input(
                                "Type Status",
                                value=current_type_status,
                                key=f"type_status_{label_fn}",
                                help="e.g., holotype, paratype, syntype"
                            )
                            if edited_type_status != original_type_status:
                                current_edits["type_status"] = edited_type_status
                            elif "type_status" in current_edits:
                                del current_edits["type_status"]
                        
                        with entity_col2:
                            st.markdown("**Occurrence & Collection**")
                            
                            # Collector
                            original_collector = entities.get("recordedBy", "")
                            current_collector = current_edits.get("recordedBy", original_collector)
                            edited_collector = st.text_input(
                                "Collector",
                                value=current_collector,
                                key=f"collector_{label_fn}",
                                help="Name(s) of person(s) who collected the specimen"
                            )
                            if edited_collector != original_collector:
                                current_edits["recordedBy"] = edited_collector
                            elif "recordedBy" in current_edits:
                                del current_edits["recordedBy"]
                            
                            # Date
                            original_date = entities.get("eventDate", "")
                            current_date = current_edits.get("eventDate", original_date)
                            edited_date = st.text_input(
                                "Collection Date",
                                value=current_date,
                                key=f"date_{label_fn}",
                                help="YYYY-MM-DD format or verbatim"
                            )
                            if edited_date != original_date:
                                current_edits["eventDate"] = edited_date
                            elif "eventDate" in current_edits:
                                del current_edits["eventDate"]
                            
                            # Institution
                            original_institution = entities.get("institutionCode", "")
                            current_institution = current_edits.get("institutionCode", original_institution)
                            edited_institution = st.text_input(
                                "Institution Code",
                                value=current_institution,
                                key=f"institution_{label_fn}",
                                help="Museum/herbarium code"
                            )
                            if edited_institution != original_institution:
                                current_edits["institutionCode"] = edited_institution
                            elif "institutionCode" in current_edits:
                                del current_edits["institutionCode"]
                        
                        # Location editing
                        st.markdown("**Geographic Information**")
                        geo = entities.get("geographic_data", {})
                        locality_col1, locality_col2 = st.columns(2)
                        
                        with locality_col1:
                            original_locality = geo.get("locality", "")
                            current_locality = current_edits.get("locality", original_locality)
                            edited_locality = st.text_area(
                                "Locality",
                                value=current_locality,
                                key=f"locality_{label_fn}",
                                height=80,
                                help="Specific location description"
                            )
                            if edited_locality != original_locality:
                                current_edits["locality"] = edited_locality
                            elif "locality" in current_edits:
                                del current_edits["locality"]
                        
                        with locality_col2:
                            original_country = geo.get("parsed", {}).get("country", "")
                            current_country = current_edits.get("country", original_country)
                            edited_country = st.text_input(
                                "Country",
                                value=current_country,
                                key=f"country_{label_fn}",
                                help="Country name"
                            )
                            if edited_country != original_country:
                                current_edits["country"] = edited_country
                            elif "country" in current_edits:
                                del current_edits["country"]
                        
                        # Show summary
                        entity_edit_count = len(current_edits)
                        if entity_edit_count > 0:
                            st.success(f"✏️ {entity_edit_count} field(s) modified for this label")
                        
                    else:
                        st.info("📭 No labels with extracted entities found. Run the pipeline or entity extraction first.")
                
                # ─── Re-run Entity Recognition ───
                st.subheader("🧬 Re-run Entity Recognition")
                st.caption(
                    "Extract structured entities (scientific names, collectors, dates, geography) "
                    "from the current OCR text using Gemini + GBIF validation + OSM geocoding. "
                    "Uses validated results if available, otherwise consolidated results."
                )
                # Determine best input source
                _er_validated_path = output_path / "validated_results.json"
                _er_consolidated_path = output_path / "consolidated_results.json"
                if _er_validated_path.exists():
                    _er_source = _er_validated_path
                    st.info(f"Source: `validated_results.json`")
                elif _er_consolidated_path.exists():
                    _er_source = _er_consolidated_path
                    st.info(f"Source: `consolidated_results.json`")
                else:
                    _er_source = None
                    st.warning("No consolidated or validated results found. Run the pipeline first.")
                
                er_opt_col1, er_opt_col2, er_opt_col3 = st.columns(3)
                with er_opt_col1:
                    er_rerun_dwc = st.checkbox("Darwin Core JSON", value=True, key="er_rerun_dwc")
                with er_opt_col2:
                    er_rerun_opends = st.checkbox("OpenDS JSON", value=False, key="er_rerun_opends")
                with er_opt_col3:
                    er_rerun_csv = st.checkbox("DwC CSV", value=False, key="er_rerun_csv")
                
                if st.button(
                    "🧬 Run Entity Recognition",
                    type="primary",
                    disabled=_er_source is None,
                    help="Extract entities with GBIF validation and OSM geocoding",
                ):
                    try:
                        from label_processing.gemini_processor import get_client as _get_client
                        from label_processing.entity_recognition import (
                            extract_and_enrich as _extract,
                            validate_and_normalize as _validate,
                            generate_dwc as _gen_dwc,
                            generate_opends as _gen_opends,
                            export_to_csv as _export_csv,
                            build_master_json as _build_master,
                        )
                        
                        with open(_er_source, "r", encoding="utf-8") as f:
                            source_data = json.load(f)
                        
                        # If source is entity_master.json-style (grouped), flatten it
                        if isinstance(source_data, list) and source_data and "labels" in source_data[0]:
                            flat = []
                            for entry in source_data:
                                for lbl in entry.get("labels", []):
                                    if "source_image" not in lbl:
                                        lbl["source_image"] = entry.get("source_image", "")
                                    flat.append(lbl)
                            source_data = flat
                        
                        label_count = len(source_data)
                        with st.spinner(f"Running entity recognition on {label_count} labels (GBIF + OSM)..."):
                            client = _get_client()
                            enriched = _extract(source_data, client)
                            validated_labels, quality_report = _validate(enriched)
                            master = _build_master(validated_labels, quality_report)
                            
                            # Save outputs
                            with open(output_path / "entity_master.json", "w", encoding="utf-8") as f:
                                json.dump(master, f, indent=2, ensure_ascii=False)
                            with open(output_path / "quality_report.json", "w", encoding="utf-8") as f:
                                json.dump(quality_report, f, indent=2, ensure_ascii=False)
                            
                            if er_rerun_dwc:
                                dwc_records = _gen_dwc(validated_labels)
                                with open(output_path / "darwin_core.json", "w", encoding="utf-8") as f:
                                    json.dump(dwc_records, f, indent=2, ensure_ascii=False)
                                if er_rerun_csv:
                                    _export_csv(dwc_records, str(output_path / "darwin_core.csv"))
                            
                            if er_rerun_opends:
                                opends_records = _gen_opends(validated_labels)
                                with open(output_path / "open_ds.json", "w", encoding="utf-8") as f:
                                    json.dump(opends_records, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"Entity recognition complete! Processed {len(validated_labels)} labels.")
                        for line in quality_report.get("summary", []):
                            st.write(f"  {line}")
                        if quality_report.get("overall_extraction_rate"):
                            st.write(f"  Overall extraction rate: {quality_report['overall_extraction_rate']}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Entity recognition failed: {e}")
                
                # ─── Result Files ───
                st.subheader("📄 Result Files")
                for file_path in result_files:
                    with st.expander(f"📄 {file_path.name}"):
                        try:
                            if file_path.suffix == '.json':
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                st.json(data)
                            elif file_path.suffix == '.csv':
                                df = pd.read_csv(file_path)
                                st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error reading {file_path.name}: {str(e)}")
                
                # ─── Downloads & Cropping ───
                st.subheader("📥 Download & Export")
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                
                with dl_col1:
                    # Download validated results (preferred) or consolidated
                    validated_path = output_path / "validated_results.json"
                    if validated_path.exists():
                        with open(validated_path, 'r') as f:
                            st.download_button(
                                "⬇️ validated_results.json",
                                f.read(),
                                file_name="validated_results.json",
                                mime="application/json",
                                key="dl_validated"
                            )
                    consolidated_path = output_path / "consolidated_results.json"
                    if consolidated_path.exists():
                        with open(consolidated_path, 'r') as f:
                            st.download_button(
                                "⬇️ consolidated_results.json",
                                f.read(),
                                file_name="consolidated_results.json",
                                mime="application/json",
                                key="dl_consolidated"
                            )
                
                with dl_col2:
                    # Download raw OCR backup
                    for name in ("ocr_gemini.json", "ocr_preprocessed.json", "ocr_google_vision.json"):
                        p = output_path / name
                        if p.exists():
                            with open(p, 'r') as f:
                                st.download_button(
                                    f"⬇️ {name}",
                                    f.read(),
                                    file_name=name,
                                    mime="application/json",
                                    key=f"dl_{name}"
                                )
                
                # ─── Entity Recognition Downloads ───
                entity_master_path = output_path / "entity_master.json"
                quality_report_path = output_path / "quality_report.json"
                dwc_path = output_path / "darwin_core.json"
                opends_path = output_path / "open_ds.json"
                csv_path = output_path / "darwin_core.csv"
                entity_files = [entity_master_path, quality_report_path, dwc_path, opends_path, csv_path]
                if any(p.exists() for p in entity_files):
                    st.subheader("🧬 Entity Recognition Outputs")
                    er_col1, er_col2, er_col3 = st.columns(3)
                    with er_col1:
                        if entity_master_path.exists():
                            with open(entity_master_path, 'r') as f:
                                st.download_button(
                                    "⬇️ entity_master.json",
                                    f.read(),
                                    file_name="entity_master.json",
                                    mime="application/json",
                                    key="dl_entity_master"
                                )
                        if quality_report_path.exists():
                            with open(quality_report_path, 'r') as f:
                                st.download_button(
                                    "⬇️ quality_report.json",
                                    f.read(),
                                    file_name="quality_report.json",
                                    mime="application/json",
                                    key="dl_quality_report"
                                )
                    with er_col2:
                        if dwc_path.exists():
                            with open(dwc_path, 'r') as f:
                                st.download_button(
                                    "⬇️ darwin_core.json",
                                    f.read(),
                                    file_name="darwin_core.json",
                                    mime="application/json",
                                    key="dl_dwc"
                                )
                        if csv_path.exists():
                            with open(csv_path, 'r') as f:
                                st.download_button(
                                    "⬇️ darwin_core.csv",
                                    f.read(),
                                    file_name="darwin_core.csv",
                                    mime="text/csv",
                                    key="dl_dwc_csv"
                                )
                    with er_col3:
                        if opends_path.exists():
                            with open(opends_path, 'r') as f:
                                st.download_button(
                                    "⬇️ open_ds.json",
                                    f.read(),
                                    file_name="open_ds.json",
                                    mime="application/json",
                                    key="dl_opends"
                                )
                
                with dl_col3:
                    # Crop images button
                    if bbox_data and st.button("✂️ Crop All Labels",
                                               help="Crop label regions from original images"):
                        crop_dir = output_path / "cropped_labels"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_count = 0
                        for img_name, labels in bbox_data.items():
                            src_path = input_path / img_name
                            if not src_path.exists():
                                continue
                            src_img = cv2.imread(str(src_path))
                            if src_img is None:
                                continue
                            for lbl in labels:
                                bbox = lbl.get("bbox", lbl)
                                x1 = max(0, int(bbox.get("xmin", 0)))
                                y1 = max(0, int(bbox.get("ymin", 0)))
                                x2 = min(src_img.shape[1], int(bbox.get("xmax", 0)))
                                y2 = min(src_img.shape[0], int(bbox.get("ymax", 0)))
                                if x2 > x1 and y2 > y1:
                                    crop = src_img[y1:y2, x1:x2]
                                    stem = Path(img_name).stem
                                    idx = lbl.get("label_index", crop_count + 1)
                                    cv2.imwrite(str(crop_dir / f"{stem}_{idx}.jpg"), crop)
                                    crop_count += 1
                        st.success(f"✂️ Cropped {crop_count} labels → `{crop_dir}`")
            else:
                st.info("📋 No result files found. Run a pipeline to generate results.")
        else:
            st.info("📁 Output directory does not exist yet.")
    else:
        st.info("🔄 Processing in progress... Results will appear when complete.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem'>"
        "🔬 <strong>ELIE</strong> · AI-powered museum specimen digitization · "
        "<a href='https://github.com/MargotBelot/entomological-label-information-extraction' "
        "style='color:#1565C0'>GitHub</a> · MIT License</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
