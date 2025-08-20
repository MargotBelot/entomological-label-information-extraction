#!/usr/bin/env python3
"""
Model Loading Troubleshooting Script
====================================

This script helps diagnose and fix model loading issues in the
entomological label information extraction pipeline.

Usage:
    python3 scripts/troubleshooting/test_model_loading.py [model_path]

If no model_path is provided, it will test the default model location.
"""

import os
import sys
import torch
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(message, success=True):
    """Print a status message with emoji."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def check_model_file(model_path):
    """Check if model file exists and is valid."""
    print_header("MODEL FILE DIAGNOSTICS")
    
    if not os.path.exists(model_path):
        print_status(f"Model file not found: {model_path}", False)
        return False
    else:
        print_status(f"Model file exists: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        print_status("Model file is empty", False)
        return False
    else:
        print_status(f"Model file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # Check file type
    import subprocess
    try:
        result = subprocess.run(['file', model_path], capture_output=True, text=True)
        print_status(f"File type: {result.stdout.strip()}")
    except:
        pass
    
    return True

def check_pytorch_version():
    """Check PyTorch installation and version."""
    print_header("PYTORCH DIAGNOSTICS")
    
    try:
        import torch
        print_status(f"PyTorch version: {torch.__version__}")
        print_status(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_status(f"CUDA version: {torch.version.cuda}")
            print_status(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print_status("PyTorch not installed", False)
        return False
    
    return True

def test_torch_loading(model_path):
    """Test loading with PyTorch directly."""
    print_header("PYTORCH LOADING TEST")
    
    try:
        # Test 1: Load normally
        print("🔍 Test 1: Standard loading...")
        model_data = torch.load(model_path)
        print_status("Standard PyTorch loading successful")
        
        if isinstance(model_data, dict):
            print_status(f"Model contains {len(model_data)} parameters")
            sample_keys = list(model_data.keys())[:5]
            print(f"   Sample keys: {sample_keys}")
        
    except Exception as e:
        print_status(f"Standard loading failed: {e}", False)
        
        # Test 2: Load with CPU mapping
        try:
            print("🔍 Test 2: CPU-mapped loading...")
            model_data = torch.load(model_path, map_location='cpu')
            print_status("CPU-mapped PyTorch loading successful")
        except Exception as e2:
            print_status(f"CPU-mapped loading failed: {e2}", False)
            
            # Test 3: Load with weights_only (newer PyTorch)
            try:
                print("🔍 Test 3: Weights-only loading...")
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                print_status("Weights-only PyTorch loading successful")
            except Exception as e3:
                print_status(f"Weights-only loading failed: {e3}", False)
                return False
    
    return True

def test_detecto_loading(model_path):
    """Test loading with Detecto."""
    print_header("DETECTO LOADING TEST")
    
    try:
        from detecto.core import Model
        print_status("Detecto imported successfully")
    except ImportError as e:
        print_status(f"Detecto import failed: {e}", False)
        return False
    
    try:
        # Test standard detecto loading
        print("🔍 Testing detecto Model.load()...")
        model = Model.load(model_path, ['label'])
        print_status("Detecto loading successful")
        return True
    except Exception as e:
        print_status(f"Detecto loading failed: {e}", False)
        return False

def test_pipeline_loading(model_path):
    """Test loading with our pipeline code."""
    print_header("PIPELINE LOADING TEST")
    
    try:
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from label_processing.label_detection import PredictLabel
        print_status("Pipeline module imported successfully")
        
        print("🔍 Testing pipeline PredictLabel loading...")
        detector = PredictLabel(model_path, ['label'])
        print_status("Pipeline loading successful")
        return True
        
    except ImportError as e:
        print_status(f"Pipeline import failed: {e}", False)
        return False
    except Exception as e:
        print_status(f"Pipeline loading failed: {e}", False)
        return False

def print_recommendations():
    """Print troubleshooting recommendations."""
    print_header("TROUBLESHOOTING RECOMMENDATIONS")
    
    print("🔧 If model loading fails, try these solutions:")
    print()
    print("1. **CUDA/CPU Mismatch Issues:**")
    print("   - Install CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("   - Or install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("2. **PyTorch Version Issues:**")
    print("   - Update PyTorch: pip install --upgrade torch torchvision")
    print("   - Check compatibility: https://pytorch.org/get-started/locally/")
    print()
    print("3. **Detecto Issues:**")
    print("   - Reinstall detecto: pip install --upgrade detecto")
    print("   - Check detecto documentation: https://detecto.readthedocs.io/")
    print()
    print("4. **File System Issues:**")
    print("   - Check file permissions: ls -la models/")
    print("   - Verify file integrity: check if model file was corrupted during transfer")
    print()
    print("5. **Environment Issues:**")
    print("   - Recreate conda environment: conda env create -f environment.yml")
    print("   - Or use virtual environment: python3 -m venv new_env && source new_env/bin/activate")

def main():
    """Main troubleshooting function."""
    print_header("ENTOMOLOGICAL LABEL EXTRACTION - MODEL LOADING DIAGNOSTICS")
    
    # Get model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default model path relative to script location
        script_dir = Path(__file__).parent
        model_path = script_dir / ".." / ".." / "models" / "label_detection_model.pth"
        model_path = str(model_path.resolve())
    
    print(f"🎯 Testing model: {model_path}")
    
    # Run diagnostics
    results = []
    results.append(("File Check", check_model_file(model_path)))
    results.append(("PyTorch Check", check_pytorch_version()))
    results.append(("PyTorch Loading", test_torch_loading(model_path)))
    results.append(("Detecto Loading", test_detecto_loading(model_path)))
    results.append(("Pipeline Loading", test_pipeline_loading(model_path)))
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    for test_name, success in results:
        print_status(f"{test_name}: {'PASSED' if success else 'FAILED'}", success)
    
    # Overall result
    all_passed = all(result[1] for result in results)
    print()
    if all_passed:
        print_status("🎉 All tests passed! Your model should work correctly.", True)
    else:
        print_status("⚠️  Some tests failed. See recommendations below.", False)
        print_recommendations()

if __name__ == "__main__":
    main()
