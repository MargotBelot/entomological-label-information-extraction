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
# Page configuration
st.set_page_config(
    page_title="ELIE - Entomological Label Information Extraction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
        # Update progress based on pipeline steps (different for MLI vs SLI)
        if 'step 1:' in output_lower:
            if pipeline_type == 'MLI' and 'detection' in output_lower:
                st.session_state.pipeline_progress = 15
                st.session_state.current_stage = "🔍 Detection"
            else:
                st.session_state.pipeline_progress = 20
                st.session_state.current_stage = "🚫 Empty/Not-Empty Classification"
        elif 'step 2:' in output_lower:
            if pipeline_type == 'MLI':
                st.session_state.pipeline_progress = 30
                st.session_state.current_stage = "🚫 Empty/Not-Empty Classification"
            else:  # SLI
                st.session_state.pipeline_progress = 40
                st.session_state.current_stage = "🏷️ ID/Description Classification"
        elif 'step 3:' in output_lower:
            if pipeline_type == 'MLI':
                st.session_state.pipeline_progress = 45
                st.session_state.current_stage = "🏷️ ID/Description Classification"
            else:  # SLI
                st.session_state.pipeline_progress = 50
                st.session_state.current_stage = "✍️ Handwritten/Printed Classification"
        elif 'step 4:' in output_lower:
            if pipeline_type == 'MLI':
                st.session_state.pipeline_progress = 60
                st.session_state.current_stage = "✍️ Handwritten/Printed Classification"
            else:  # SLI
                st.session_state.pipeline_progress = 65
                st.session_state.current_stage = "🔄 Rotation Correction"
        elif 'step 5:' in output_lower:
            if pipeline_type == 'MLI':
                st.session_state.pipeline_progress = 75
                st.session_state.current_stage = "📖 OCR Processing"
            else:  # SLI
                st.session_state.pipeline_progress = 80
                st.session_state.current_stage = "📖 OCR Processing"
        elif 'step 6:' in output_lower:
            if pipeline_type == 'MLI':
                st.session_state.pipeline_progress = 90
                st.session_state.current_stage = "🔧 Post-processing"
            else:  # SLI
                st.session_state.pipeline_progress = 90
                st.session_state.current_stage = "🔧 Post-processing"
        elif 'pipeline completed successfully' in output_lower or '✅ pipeline completed successfully' in output_lower:
            st.session_state.pipeline_progress = 100
            st.session_state.current_stage = "✅ Completed"
            st.session_state.processing = False
        
    def get_input_images(self, input_dir):
        """Get list of input images"""
        input_path = Path(input_dir)
        if not input_path.exists():
            return []
        
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
            else:
                script_path = self.project_root / "tools" / "pipelines" / "run_sli_pipeline_conda.sh"
            
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
    st.title("🔬 ELIE - Entomological Label Information Extraction")
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
            ["MLI", "SLI"],
            help="MLI: Multi-Label (full specimen photos), SLI: Single-Label (pre-cropped labels)"
        )
        
        # Input directory
        default_input = str(Path(__file__).parent.parent / "data" / pipeline_type / "input")
        input_dir = st.text_input(
            "Input Directory", 
            value=default_input,
            help="Directory containing images to process"
        )
    
    with col2:
        # Output directory
        default_output = str(Path(__file__).parent.parent / "data" / pipeline_type / "output")
        output_dir = st.text_input(
            "Output Directory",
            value=default_output,
            help="Directory where results will be saved"
        )
        
        # Processing options
        batch_size = st.slider("Batch Size", 1, 8, 1, help="Number of images to process simultaneously")
    
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
                            st.image(img, caption=img_path.name, width='stretch')
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
            # Images processed (estimate)
            if 'images' in locals():
                processed_images = int((overall_progress / 100) * len(images))
                st.metric("Images Processed", f"{processed_images}/{len(images)}")
        
        
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
                else:
                    script_path = st.session_state.processor.project_root / "tools" / "pipelines" / "run_sli_pipeline_conda.sh"
                
                # Validate input directory first
                input_path = Path(input_dir)
                if not input_path.exists():
                    st.session_state.logs.append(f"[{current_time}] ❌ Input directory does not exist: {input_dir}")
                    st.session_state.processing = False
                    return
                
                # Check for images in input directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                input_images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
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
        if output_path.exists():
            # Look for result files
            result_files = list(output_path.glob("*.json")) + list(output_path.glob("*.csv"))
            image_files = list(output_path.glob("**/*.jpg")) + list(output_path.glob("**/*.png"))
            
            if result_files or image_files:
                st.success(f"✅ Found {len(result_files)} result files and {len(image_files)} images")
                
                # Simple file browser
                if result_files:
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
                                    st.dataframe(df, width='stretch')
                            except Exception as e:
                                st.error(f"❌ Error reading {file_path.name}: {str(e)}")
                
                if image_files:
                    st.subheader("🖼️ Generated Images")
                    cols = st.columns(3)
                    for i, img_path in enumerate(image_files[:9]):  # Show first 9 images
                        with cols[i % 3]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=img_path.name, width='stretch')
                            except Exception as e:
                                st.error(f"❌ Error loading {img_path.name}")
            else:
                st.info("📋 No result files found. Run a pipeline to generate results.")
        else:
            st.info("📁 Output directory does not exist yet.")
    else:
        st.info("🔄 Processing in progress... Results will appear when complete.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **ELIE** - AI-powered system for museum specimen digitization | "
        "[GitHub](https://github.com/MargotBelot/entomological-label-information-extraction) | "
        "License: MIT"
    )

if __name__ == "__main__":
    main()
