#!/usr/bin/env python3
"""
Configuration module for entomological label information extraction.
Handles platform-specific paths and environment variables.
"""

import os
import sys
from pathlib import Path
from typing import Dict


IMAGE_EXTENSIONS = ("jpg", "jpeg")


class PathConfig:
    """
    Centralized path configuration for cross-platform compatibility.
    """

    def __init__(self):
        """Initialize path configuration."""
        self.project_root = self._get_project_root()
        self.platform = sys.platform
        self._setup_paths()

    def _get_project_root(self) -> Path:
        """
        Get the project root directory dynamically.

        Returns:
            Path: Absolute path to project root
        """
        # Try to find project root by looking for key files
        current_dir = Path(__file__).parent.absolute()

        # Look for characteristic files that indicate project root
        indicators = ["environment.yml", "README.md", "models", "label_processing"]

        for parent in [current_dir] + list(current_dir.parents):
            if all((parent / indicator).exists() for indicator in indicators[:2]):
                return parent

        # Fallback to current directory
        return current_dir

    def _setup_paths(self):
        """Setup all project paths."""
        # Base directories
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "output"
        self.test_data_dir = self.project_root / "unit_tests" / "testdata"

        # Model paths
        self.detection_model_path = self.models_dir / "label_detection_model.pth"
        self.classifier_models = {
            "identifier": self.models_dir
            / "label_classifier_identifier_not_identifier",
            "handwritten_printed": self.models_dir / "label_classifier_hp",
            "multi_single": self.models_dir / "label_classifier_multi_single",
        }

        # Class files
        self.classes_dir = self.models_dir / "classes"
        self.class_files = {
            "hp": self.classes_dir / "hp_classes.txt",
            "ms": self.classes_dir / "ms_classes.txt",
            "nuri": self.classes_dir / "nuri_classes.txt",
        }

        # Environment-specific overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides for paths."""
        # Allow override of project root
        env_project_root = os.getenv("ENTOMOLOGICAL_PROJECT_ROOT")
        if env_project_root:
            self.project_root = Path(env_project_root)
            self._setup_paths()  # Recalculate paths with new root

        # Allow override of models directory
        env_models_dir = os.getenv("ENTOMOLOGICAL_MODELS_DIR")
        if env_models_dir:
            self.models_dir = Path(env_models_dir)
            self._update_model_paths()

        # Allow override of specific model files
        env_detection_model = os.getenv("ENTOMOLOGICAL_DETECTION_MODEL_PATH")
        if env_detection_model:
            self.detection_model_path = Path(env_detection_model)

    def _update_model_paths(self):
        """Update model paths when models directory changes."""
        self.detection_model_path = self.models_dir / "label_detection_model.pth"
        self.classifier_models = {
            "identifier": self.models_dir
            / "label_classifier_identifier_not_identifier",
            "handwritten_printed": self.models_dir / "label_classifier_hp",
            "multi_single": self.models_dir / "label_classifier_multi_single",
        }
        self.classes_dir = self.models_dir / "classes"
        self.class_files = {
            "hp": self.classes_dir / "hp_classes.txt",
            "ms": self.classes_dir / "ms_classes.txt",
            "nuri": self.classes_dir / "nuri_classes.txt",
        }

    def get_model_path(self, model_type: str) -> Path:
        """
        Get path for a specific model type.

        Args:
            model_type: Type of model ('detection', 'identifier', 'handwritten_printed', 'multi_single')

        Returns:
            Path: Path to the model file

        Raises:
            ValueError: If model type is not recognized
        """
        if model_type == "detection":
            return self.detection_model_path
        elif model_type in self.classifier_models:
            return self.classifier_models[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_class_names(self, model_type: str) -> list:
        """
        Get class names for a specific model type.

        Args:
            model_type: Type of model ('identifier', 'handwritten_printed', 'multi_single')

        Returns:
            list: List of class names
        """
        class_mappings = {
            "identifier": ["not_identifier", "identifier"],
            "handwritten_printed": ["handwritten", "printed"],
            "multi_single": ["multi", "single"],
        }
        return class_mappings.get(model_type, [])

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.models_dir,
            self.data_dir,
            self.output_dir,
            self.classes_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all required paths exist.

        Returns:
            Dict[str, bool]: Dictionary mapping path names to existence status
        """
        validation_results = {}

        # Check directories
        validation_results["project_root"] = self.project_root.exists()
        validation_results["models_dir"] = self.models_dir.exists()
        validation_results["classes_dir"] = self.classes_dir.exists()

        # Check model files (optional, as they might not exist in fresh installs)
        validation_results["detection_model"] = self.detection_model_path.exists()

        for model_name, model_path in self.classifier_models.items():
            validation_results[f"classifier_{model_name}"] = model_path.exists()

        # Check class files
        for class_name, class_file in self.class_files.items():
            validation_results[f"classes_{class_name}"] = class_file.exists()

        return validation_results

    def get_temp_dir(self) -> Path:
        """
        Get a temporary directory for the current platform.

        Returns:
            Path: Platform-appropriate temporary directory
        """
        if self.platform.startswith("win"):
            temp_base = Path(os.getenv("TEMP", "C:/temp"))
        else:
            temp_base = Path("/tmp")

        temp_dir = temp_base / "entomological_extraction"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""
PathConfig for {self.platform}:
  Project Root: {self.project_root}
  Models Dir: {self.models_dir}
  Detection Model: {self.detection_model_path}
  Output Dir: {self.output_dir}
        """.strip()


# Global configuration instance
config = PathConfig()


# Convenience functions
def get_project_root() -> Path:
    """Get the project root directory."""
    return config.project_root


def get_model_path(model_type: str) -> Path:
    """Get path for a specific model."""
    return config.get_model_path(model_type)


def get_models_dir() -> Path:
    """Get the models directory."""
    return config.models_dir


def get_output_dir() -> Path:
    """Get the output directory."""
    return config.output_dir


def validate_setup() -> bool:
    """
    Validate the current setup.

    Returns:
        bool: True if setup is valid, False otherwise
    """
    validation_results = config.validate_paths()

    # Print validation results
    print("Path Validation Results:")
    for path_name, exists in validation_results.items():
        status = "" if exists else ""
        print(f"  {status} {path_name}: {exists}")

    # Check if critical paths exist
    critical_paths = ["project_root", "models_dir"]
    return all(validation_results.get(path, False) for path in critical_paths)


if __name__ == "__main__":
    # Print configuration when run directly
    print(config)
    print("\nValidating setup...")
    is_valid = validate_setup()
    print(f"\nSetup is {'valid' if is_valid else 'invalid'}")
