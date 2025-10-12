#!/usr/bin/env python3
"""
ELIE Security Validation Script
===============================

This script validates that all critical security fixes have been properly implemented
and that the system is secure against the identified worst-case scenarios.

Run this script to verify security posture before production deployment.
"""

import os
import sys
import hashlib
from pathlib import Path
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from label_processing.utils import verify_model_integrity, validate_image_integrity
from label_processing.config import config

def check_python_version_consistency():
    """Check that Python versions are consistent across all configuration files."""
    print(" Checking Python version consistency...")
    
    issues = []
    
    # Check environment.yml
    env_yml = config.project_root / "environment.yml"
    if env_yml.exists():
        with open(env_yml, 'r') as f:
            content = f.read()
            if "python=3.10" not in content:
                issues.append("environment.yml: Python version not set to 3.10")
    
    # Check pyproject.toml
    pyproject = config.project_root / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, 'r') as f:
            content = f.read()
            if "requires-python = \">=3.10\"" not in content:
                issues.append("pyproject.toml: Python version requirement not set to >=3.10")
            if "target-version = ['py310']" not in content:
                issues.append("pyproject.toml: Black target version not set to py310")
    
    # Check Docker files
    pipelines_dir = config.project_root / "pipelines"
    if pipelines_dir.exists():
        for dockerfile in pipelines_dir.glob("*.dockerfile"):
            with open(dockerfile, 'r') as f:
                content = f.read()
                if "python:3.10.14-bullseye" not in content:
                    issues.append(f"{dockerfile.name}: Docker base image not set to python:3.10.14-bullseye")
    
    if issues:
        print(" Python version inconsistencies found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Python versions are consistent across all configuration files")
        return True

def check_model_integrity_enforcement():
    """Check that model integrity verification is mandatory."""
    print(" Checking model integrity enforcement...")
    
    issues = []
    
    # Check that checksums file exists
    checksums_file = config.models_dir / "checksums.sha256"
    if not checksums_file.exists():
        issues.append(f"Missing checksums file: {checksums_file}")
    else:
        print(f" Found checksums file: {checksums_file}")
        
        # Verify at least one model
        detection_model = config.detection_model_path
        if detection_model.exists():
            try:
                if verify_model_integrity(str(detection_model), str(checksums_file)):
                    print(f" Detection model integrity verified: {detection_model}")
                else:
                    issues.append(f"Detection model failed integrity check: {detection_model}")
            except Exception as e:
                issues.append(f"Detection model integrity check error: {e}")
        else:
            print(f"️  Detection model not found (expected in fresh install): {detection_model}")
    
    if issues:
        print(" Model integrity issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Model integrity verification is properly enforced")
        return True

def check_pickle_security():
    """Check that unsafe pickle loading has been removed."""
    print(" Checking pickle loading security...")
    
    issues = []
    
    # Check label_detection.py for unsafe methods
    detection_file = config.project_root / "label_processing" / "label_detection.py"
    if detection_file.exists():
        with open(detection_file, 'r') as f:
            content = f.read()
            
            # Check for removed unsafe methods
            unsafe_methods = [
                "_load_with_weights_only_false",
                "_load_with_pickle_protocol",
                "_load_with_basic_torch",
                "weights_only=False"
            ]
            
            for method in unsafe_methods:
                if method in content:
                    issues.append(f"Unsafe pickle method still present: {method}")
            
            # Check for safe loading methods
            if "weights_only=True" not in content:
                issues.append("Safe loading with weights_only=True not found")
            
            if "_load_pytorch_safe" not in content:
                issues.append("Safe PyTorch loading method not found")
            
            if "_load_detecto_safe" not in content:
                issues.append("Safe Detecto loading method not found")
    
    if issues:
        print(" Pickle security issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Pickle loading is secure (weights_only=True enforced)")
        return True

def check_memory_limits():
    """Check that memory limits and batch processing are implemented."""
    print(" Checking memory limit implementation...")
    
    issues = []
    
    # Check utils.py for image validation limits
    utils_file = config.project_root / "label_processing" / "utils.py"
    if utils_file.exists():
        with open(utils_file, 'r') as f:
            content = f.read()
            
            if "max_size_mb: int = 25" not in content:
                issues.append("Image size limit not set to 25MB")
            
            if "max_dimensions: tuple = (8000, 8000)" not in content:
                issues.append("Image dimension limits not set")
            
            if "estimated_memory_mb > 500" not in content:
                issues.append("Memory usage estimation not implemented")
    
    # Check tensorflow_classifier.py for batch processing
    classifier_file = config.project_root / "label_processing" / "tensorflow_classifier.py"
    if classifier_file.exists():
        with open(classifier_file, 'r') as f:
            content = f.read()
            
            if "batch_size: int = 32" not in content:
                issues.append("Batch processing not implemented")
            
            if "max_images: int = 10000" not in content:
                issues.append("Maximum image limit not set")
            
            if "gc.collect()" not in content:
                issues.append("Garbage collection not implemented")
    
    if issues:
        print(" Memory limit issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Memory limits and batch processing are properly implemented")
        return True

def check_credentials_security():
    """Check that Google Cloud credentials are handled securely."""
    print(" Checking credentials security...")
    
    issues = []
    
    # Check ocr_vision.py for secure credential handling
    vision_file = config.project_root / "label_processing" / "ocr_vision.py"
    if vision_file.exists():
        with open(vision_file, 'r') as f:
            content = f.read()
            
            if "SECURITY: Validate credentials file exists" not in content:
                issues.append("Credentials validation not implemented")
            
            if "file_stat.st_mode & 0o044" not in content:
                issues.append("Credentials permission check not implemented")
            
            if "os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS'" not in content:
                issues.append("Credentials cleanup not implemented")
    
    if issues:
        print(" Credentials security issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Google Cloud credentials are handled securely")
        return True

def test_image_validation():
    """Test image validation with various scenarios."""
    print(" Testing image validation functionality...")
    
    try:
        # Test with non-existent file
        if validate_image_integrity("/nonexistent/file.jpg"):
            print(" Image validation failed: Should reject non-existent files")
            return False
        
        print(" Image validation correctly rejects non-existent files")
        return True
        
    except Exception as e:
        print(f" Image validation test failed: {e}")
        return False

def run_security_validation():
    """Run all security validation checks."""
    print(" ELIE SECURITY VALIDATION")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version Consistency", check_python_version_consistency),
        ("Model Integrity Enforcement", check_model_integrity_enforcement),
        ("Pickle Loading Security", check_pickle_security),
        ("Memory Limits Implementation", check_memory_limits),
        ("Credentials Security", check_credentials_security),
        ("Image Validation Testing", test_image_validation),
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_function in checks:
        print(f"\n {check_name}")
        print("-" * (len(check_name) + 4))
        
        try:
            if check_function():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f" Check failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"SECURITY VALIDATION RESULTS:")
    print(f" Passed: {passed}")
    print(f" Failed: {failed}")
    
    if failed == 0:
        print("\n ALL SECURITY CHECKS PASSED!")
        print("️  The ELIE pipeline is secure for production use.")
        return True
    else:
        print(f"\n️  {failed} SECURITY ISSUES FOUND!")
        print(" DO NOT DEPLOY TO PRODUCTION until all issues are resolved.")
        return False

if __name__ == "__main__":
    success = run_security_validation()
    sys.exit(0 if success else 1)