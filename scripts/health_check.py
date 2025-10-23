#!/usr/bin/env python3
"""
Health Check Script for Entomological Label Information Extraction
Validates system requirements and provides diagnostic information.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check Python version and provide recommendations."""
    version = sys.version_info
    platform = sys.platform
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 11):
        print("[OK] Excellent! Python 3.11+ detected")
    elif version >= (3, 10):
        print("[OK] Good! Python 3.10+ detected")
    elif version >= (3, 9):
        print("[WARNING] Python 3.9 detected. 3.10+ recommended for full compatibility")
        if platform == "darwin":  # macOS
            print("[HINT] macOS: brew install python@3.11")
        elif platform.startswith("linux"):
            print("[HINT] Linux: sudo apt install python3.11 (Ubuntu/Debian) or equivalent")
        elif platform.startswith("win"):
            print("[HINT] Windows: Download from python.org or use Microsoft Store")
        else:
            print("[HINT] Visit python.org for installation instructions")
    else:
        print("[ERROR] Python version too old. 3.10+ required")
        return False
    return True

def check_docker():
    """Check Docker installation and status."""
    try:
        # Check if Docker is installed
        docker_path = shutil.which("docker")
        if not docker_path:
            print("Docker: [ERROR] Not installed")
            platform = sys.platform
            if platform == "darwin":  # macOS
                print("[HINT] macOS: Install Docker Desktop from https://docker.com")
            elif platform.startswith("linux"):
                print("[HINT] Linux: sudo apt install docker.io (Ubuntu/Debian) or visit https://docker.com")
            elif platform.startswith("win"):
                print("[HINT] Windows: Install Docker Desktop from https://docker.com")
            else:
                print("[HINT] Visit https://docker.com for installation instructions")
            return False
        
        # Check Docker version
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Docker: {version}")
        
        # Check if Docker daemon is running
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Docker daemon is running")
            return True
        else:
            print("[WARNING] Docker installed but daemon not running")
            platform = sys.platform
            if platform == "darwin"or platform.startswith("win"):  # macOS or Windows
                print("[HINT] Start Docker Desktop application")
            elif platform.startswith("linux"):
                print("[HINT] Linux: sudo systemctl start docker")
            else:
                print("[HINT] Start Docker service for your platform")
            return False
            
    except Exception as e:
        print(f"Docker: [ERROR] Error checking Docker: {e}")
        return False

def check_project_structure():
    """Check if we're in the correct project directory."""
    try:
        current_dir = Path.cwd()
        project_name = "entomological-label-information-extraction"
        
        print(f"Project: {current_dir.name}")
    except Exception as e:
        print(f"Project: [WARNING] Could not determine current directory: {e}")
        # Try to get script directory instead
        script_dir = Path(__file__).parent.parent
        current_dir = script_dir
        print(f"Using script directory: {current_dir.name}")
    
    required_files = [
        "launch_gui.py",
        "pyproject.toml",
        "environment.yml", 
        "README.md"
    ]
    
    required_dirs = [
        "data",
        "docs", 
        "scripts",
        "tools",
        "models",
        "pipelines"
    ]
    
    missing_files = [f for f in required_files if not (current_dir / f).exists()]
    missing_dirs = [d for d in required_dirs if not (current_dir / d).exists()]
    
    if not missing_files and not missing_dirs:
        print("[OK] All required files and directories present")
        return True
    else:
        print("[WARNING] Missing components:")
        for f in missing_files:
            print(f"- {f}")
        for d in missing_dirs:
            print(f"- {d}/")
        return False

def check_system_resources():
    """Check available system resources."""
    try:
        # Check available disk space - cross-platform approach
        # Try current directory first, fallback to script directory
        check_path = '.'
        try:
            os.statvfs('.')  # Test if current directory is accessible
        except:
            check_path = os.path.dirname(os.path.abspath(__file__))
            
        if sys.platform.startswith('win'):
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(check_path), ctypes.pointer(free_bytes), None, None)
            free_gb = free_bytes.value / (1024**3)
        else:
            # Unix-like systems (Linux, macOS)
            statvfs = os.statvfs(check_path)
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            free_gb = free_bytes / (1024**3)
        
        print(f"Disk Space: {free_gb:.1f}GB available")
        
        if free_gb < 5:
            print("[WARNING] Low disk space. 5GB+ recommended")
        else:
            print("[OK] Sufficient disk space")
            
        return True
    except Exception as e:
        print(f"Disk Space: [WARNING] Could not check disk space: {e}")
        return True

def check_dependencies():
    """Check for optional dependencies."""
    deps = {
        "git": "Version control",
        "conda": "Package manager (optional)",
        "tesseract": "OCR engine (for local processing)"
    }
    
    print("Dependencies:")
    all_good = True
    
    for dep, description in deps.items():
        path = shutil.which(dep)
        if path:
            try:
                result = subprocess.run([dep, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.split('\n')[0]
                    print(f"[OK] {dep}: {version}")
                else:
                    print(f"[OK] {dep}: Installed")
            except:
                print(f"[OK] {dep}: Installed")
        else:
            if dep == "tesseract":
                print(f"[WARNING] {dep}: Not found ({description})")
                platform = sys.platform
                if platform == "darwin":  # macOS
                    print("[HINT] macOS: brew install tesseract")
                elif platform.startswith("linux"):
                    print("[HINT] Linux: sudo apt install tesseract-ocr (Ubuntu/Debian)")
                elif platform.startswith("win"):
                    print("[HINT] Windows: Download from https://github.com/tesseract-ocr/tesseract")
                else:
                    print("[HINT] Visit https://github.com/tesseract-ocr/tesseract")
            elif dep == "conda":
                print(f"[INFO] {dep}: Not found ({description})")
            else:
                print(f"[WARNING] {dep}: Not found ({description})")
                all_good = False
    
    return all_good

def main():
    """Run comprehensive health check."""
    print("Health Check - Entomological Label Information Extraction")
    print("="* 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Docker", check_docker),
        ("Project Structure", check_project_structure),
        ("System Resources", check_system_resources),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print()
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] Error during {name} check: {e}")
            results.append((name, False))
    
    print()
    print("="* 60)
    print("Health Check Summary:")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "[PASS]"if result else "[FAIL]"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print()
    if passed == total:
        print("[SUCCESS] All checks passed! Your system is ready to run the extraction tool.")
        print("Ready to launch GUI: python3 launch_gui.py")
        print("Or run command line: ./tools/run_mli_pipeline.sh or ./tools/run_sli_pipeline.sh")
    elif passed >= total - 1:
        print("[MOSTLY OK] Most checks passed. You should be able to run the tool with Docker.")
        print("Ready to launch GUI: python3 launch_gui.py")
        print("Or run command line: ./tools/run_mli_pipeline.sh or ./tools/run_sli_pipeline.sh")
    else:
        print("[ERROR] Several checks failed. Please resolve the issues above before running.")
        print("See README.md for detailed setup instructions.")
    
    print()
    print("For help:")
    print("README.md - Complete installation and usage guide")
    print("python launch_gui.py - Start the graphical interface")

if __name__ == "__main__":
    main()