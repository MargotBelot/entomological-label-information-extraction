# Import third-party libraries
import cv2
import torch
import os
import glob
import detecto.utils
import multiprocessing as mp
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
import sys
from detecto.core import Model
import pickle

from typing import Union

import warnings
import platform

# Suppress torchvision deprecation warnings from detecto library
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# ---------------------Image Segmentation---------------------#

# --- START: added image-file helpers and small robustness fixes ---
# helper: only try real image files
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def is_image_file(path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    name = p.name
    if name.startswith("._") or name.startswith("."):
        return False
    return p.suffix.lower() in IMAGE_EXTS


# --- END: added helpers ---


class PredictLabel:
    """
    Class for predicting labels using a trained object detection model.

    Attributes:
        path_to_model (str): Path to the trained model file.
        classes (list): List of classes used in the model.
        jpg_path (str|Path|None): Path to a specific JPG file for prediction.
        threshold (float): Threshold value for scores. Defaults to 0.8.
        model (detecto.core.Model): Trained object detection model.
    """

    def __init__(
        self,
        path_to_model: str,
        classes: list,
        jpg_path: Union[str, Path, None] = None,
        threshold: float = 0.8,
    ) -> None:
        """
        Init Method for the PredictLabel Class.

        Args:
            path_to_model (str): Path to the model.
            classes (list): List of classes.
            jpg_path (str|Path|None): Path to JPG file for prediction.
            threshold (float, optional): Threshold value for scores.
        """
        self.path_to_model = path_to_model
        self.classes = classes
        self.jpg_path = jpg_path
        self.threshold = threshold
        self.model = self.retrieve_model()

    @property
    def jpg_path(self):
        """str|Path|None: Property for JPG path."""
        return self._jpg_path

    @jpg_path.setter
    def jpg_path(self, jpg_path: Union[str, Path]):
        """Setter for JPG path."""
        if jpg_path == None:
            self._jpg_path = None
        elif isinstance(jpg_path, str):
            self._jpg_path = Path(jpg_path)
        elif isinstance(jpg_path, Path):
            self._jpg_path = jpg_path

    def retrieve_model(self) -> detecto.core.Model:
        """
        Retrieve the trained object detection model using Detecto's Model.load.
        Includes cross-platform compatibility fixes and integrity verification.
        """
        if not os.path.exists(self.path_to_model):
            raise FileNotFoundError(f"Model file '{self.path_to_model}' not found.")
        if os.path.getsize(self.path_to_model) == 0:
            raise IOError(f"Model file '{self.path_to_model}' is empty.")

        # Verify model integrity if checksums file exists
        model_dir = os.path.dirname(self.path_to_model)
        checksums_file = os.path.join(model_dir, "checksums.sha256")
        if os.path.exists(checksums_file):
            try:
                from label_processing.utils import verify_model_integrity

                if not verify_model_integrity(self.path_to_model, checksums_file):
                    print(
                        f"WARNING: Model integrity check failed for {self.path_to_model}"
                    )
                else:
                    print(f"Model integrity verified for {self.path_to_model}")
            except Exception as e:
                print(f"Could not verify model integrity: {e}")

        print("Loading model from:", self.path_to_model)

        # Set environment for cross-platform compatibility
        self._setup_cross_platform_environment()

        # SECURITY: Only use safe loading strategies with weights_only=True
        loading_strategies = [
            # Strategy 1: SAFE PyTorch loading with mandatory weights_only=True
            lambda: self._load_pytorch_safe(),
            # Strategy 2: SAFE detecto loading with verification
            lambda: self._load_detecto_safe(),
        ]

        last_error = None
        for i, strategy in enumerate(loading_strategies, 1):
            try:
                print(f"Trying loading strategy {i}...")
                model = strategy()
                print("Model loaded successfully")
                return model
            except Exception as e:
                print(f"Strategy {i} failed: {e}")
                last_error = e
                continue

        # If all strategies fail, raise the last error
        print(f"All loading strategies failed. Last error: {last_error}")
        raise last_error

    def _setup_cross_platform_environment(self):
        """Setup environment variables for cross-platform compatibility."""
        import platform

        # Force CPU-only execution to avoid CUDA issues on Linux servers
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Set multiprocessing start method for Linux compatibility
        if platform.system() == "Linux":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Method already set, ignore
                pass

        # Set PyTorch thread limits for stable performance
        torch.set_num_threads(1)

        # Disable MKL optimizations that can cause issues on some Linux distributions
        if platform.system() == "Linux":
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def _load_pytorch_safe(self):
        """SECURITY: Safe PyTorch loading with mandatory weights_only=True."""
        try:
            print("SECURITY: Attempting SAFE PyTorch loading with weights_only=True")

            # SECURITY: Always use weights_only=True to prevent code injection
            state_dict = torch.load(
                self.path_to_model, map_location="cpu", weights_only=True
            )

            # Create a new model instance and load state dict safely
            model = Model(self.classes)
            if hasattr(model, "model"):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            return model

        except Exception as e:
            print(f"SECURITY: Safe PyTorch loading failed: {e}")
            raise Exception(
                f"SECURITY ERROR: Could not load model safely. "
                f"Model may be corrupted or use unsafe pickle objects. "
                f"Error: {e}"
            )

    def _load_detecto_safe(self):
        """SECURITY: Safe detecto loading with integrity verification."""
        try:
            print("SECURITY: Attempting SAFE detecto loading with verification")

            # SECURITY: Verify model integrity before loading
            model_dir = os.path.dirname(self.path_to_model)
            checksums_file = os.path.join(model_dir, "checksums.sha256")

            if not os.path.exists(checksums_file):
                raise Exception(
                    f"SECURITY ERROR: No checksums file found at {checksums_file}. "
                    f"Model integrity cannot be verified."
                )

            from label_processing.utils import verify_model_integrity

            if not verify_model_integrity(self.path_to_model, checksums_file):
                raise Exception(
                    f"SECURITY ERROR: Model integrity verification failed for {self.path_to_model}. "
                    f"Model may be corrupted or tampered with."
                )

            # Only load if integrity is verified
            print("SECURITY: Model integrity verified, proceeding with safe loading")

            # Monkey-patch torch.load to enforce weights_only=True
            original_torch_load = torch.load

            def safe_patched_load(*args, **kwargs):
                kwargs["weights_only"] = True  # SECURITY: Force safe loading
                return original_torch_load(*args, **kwargs)

            torch.load = safe_patched_load

            try:
                model = Model.load(self.path_to_model, self.classes)
                return model
            finally:
                torch.load = original_torch_load

        except Exception as e:
            print(f"SECURITY: Safe detecto loading failed: {e}")
            raise Exception(
                f"SECURITY ERROR: Could not load model safely via detecto. "
                f"Model integrity verification failed or model uses unsafe objects. "
                f"Error: {e}"
            )

    def class_prediction(self, jpg_path: Path = None) -> pd.DataFrame:
        """
        Predict labels for a given JPG file.

        Args:
            jpg_path (Path): Path to the JPG file.

        Returns:
            pd.DataFrame: Pandas DataFrame with prediction results.
        """
        if jpg_path is None:
            jpg_path = self.jpg_path

        # Validate the requested path
        if jpg_path is None:
            return pd.DataFrame()

        jpg_path = Path(jpg_path)
        if not is_image_file(jpg_path):
            print(f"Skipping non-image or hidden file: {jpg_path}")
            return pd.DataFrame()

        try:
            image = detecto.utils.read_image(str(jpg_path))
        except Exception as e:
            print(f"Skipping unreadable image {jpg_path}: {e}")
            return pd.DataFrame()

        try:
            predictions = self.model.predict(image)
        except Exception as e:
            print(f"Prediction failed for {jpg_path}: {e}")
            return pd.DataFrame()

        labels, boxes, scores = predictions

        entries = []
        for i, labelname in enumerate(labels):
            entry = {}
            entry["filename"] = jpg_path.name
            entry["class"] = labelname
            entry["score"] = scores[i].item()
            entry["xmin"] = boxes[i][0]
            entry["ymin"] = boxes[i][1]
            entry["xmax"] = boxes[i][2]
            entry["ymax"] = boxes[i][3]
            entries.append(entry)
        return pd.DataFrame(entries)


def prediction_parallel(
    jpg_dir: Union[str, Path], predictor: PredictLabel, n_processes: int
) -> pd.DataFrame:
    """
    Perform predictions for all JPG files in a directory with parallel processing.

    Args:
        jpg_dir (Path|str): Path to JPG files for prediction.
        predictor (PredictLabel): Prediction instance.
        n_processes (int): Number of processes for parallel execution.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the predictions.
    """
    if not isinstance(jpg_dir, Path):
        jpg_dir = Path(jpg_dir)

    # Collect image files while skipping hidden and macOS '._*' files
    file_names: list[Path] = [p for p in sorted(jpg_dir.iterdir()) if is_image_file(p)]

    # Validate readability with cv2 (some files can exist but be corrupted)
    valid_files = []
    for file in file_names:
        img = cv2.imread(str(file))
        if img is None:
            print(f"Skipping corrupted or unreadable image: {file}")
        else:
            valid_files.append(file)

    mp.set_start_method("spawn", force=True)
    with mp.Pool(n_processes) as executor:
        results = list(executor.map(predictor.class_prediction, valid_files))

    # filter empty DataFrames and concatenate if any results exist
    results = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def clean_predictions(
    jpg_dir: Path, dataframe: pd.DataFrame, threshold: float, out_dir=None
) -> pd.DataFrame:
    """
    Filter predictions based on a threshold and save the results to a CSV file.

    Args:
        jpg_dir (Path): Path to the directory with JPG files.
        dataframe (pd.DataFrame): Pandas DataFrame with predictions.
        threshold (float): Threshold value for scores.
        out_dir (str): Output directory for saving the CSV file.

    Returns:
        pd.DataFrame: Pandas DataFrame with filtered results.
    """
    # Ensure jpg_dir is a Path object
    jpg_dir = Path(jpg_dir)

    print("\nFilter coordinates")
    colnames = ["score", "xmin", "ymin", "xmax", "ymax"]
    for header in colnames:
        dataframe[header] = (
            dataframe[header]
            .astype("str")
            .str.extractall(r"(\d+\.\d+)")
            .unstack()
            .fillna("")
            .sum(axis=1)
            .astype(float)
        )
    dataframe = dataframe.loc[dataframe["score"] >= threshold]
    dataframe[["xmin", "ymin", "xmax", "ymax"]] = dataframe[
        ["xmin", "ymin", "xmax", "ymax"]
    ].fillna("0")

    if out_dir is None:
        parent_dir = jpg_dir.resolve().parent
    else:
        parent_dir = out_dir
    filename = f"{jpg_dir.stem}_predictions.csv"
    csv_path = f"{parent_dir}/{filename}"
    dataframe.to_csv(csv_path)
    print(f"\nThe csv_file {filename} has been successfully saved in {out_dir}")
    return dataframe


# ---------------------Image Cropping---------------------#


def crop_picture(img_raw: np.ndarray, path: str, filename: str, **coordinates) -> None:
    """
    Crop the picture using the given coordinates.

    Args:
        img_raw (numpy.ndarray): Input JPG converted to a numpy matrix by cv2.
        path (str): Path where the picture should be saved.
        filename (str): Name of the picture.
        coordinates: Coordinates for cropping.
    """
    xmin = coordinates["xmin"]
    ymin = coordinates["ymin"]
    xmax = coordinates["xmax"]
    ymax = coordinates["ymax"]
    filepath = f"{path}/{filename}"
    crop = img_raw[ymin:ymax, xmin:xmax]
    cv2.imwrite(filepath, crop)


def create_crops(
    jpg_dir: Path, dataframe: pd.DataFrame, out_dir: Union[str, Path] = Path(os.getcwd())
) -> None:
    """
    Creates crops by using the csv from applying the model and the original
    pictures inside a directory.

    Args:
        jpg_dir (): path to directory with jpgs.
        dataframe (str): path to csv file.
        out_dir (Path): path to the target directory to save the cropped jpgs.
    """
    dir_path = jpg_dir
    out_dir = Path(out_dir)
    new_dir_name = Path(dir_path).name + "_cropped"
    path = out_dir.joinpath(new_dir_name)
    path.mkdir(parents=True, exist_ok=True)

    total_crops = 0
    # iterate Path objects and skip hidden / '._*' files
    for p in sorted(Path(dir_path).glob("*.jpg")):
        filepath = str(p)
        if not p.exists():
            print(f"File cannot be found: {filepath}")
            continue
        if not is_image_file(p):
            print(f"Skipping hidden or non-image file: {filepath}")
            continue

        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]

        if match.empty:
            print(f"No predictions for image: {filename}. Skipping...")
            continue

        image_raw = cv2.imread(filepath)
        if image_raw is None:
            print(f"Error: Impossible to read the image {filepath}. Corrupted file?")
            continue

        label_id = Path(filename).stem
        label_occ = []
        for _, row in match.iterrows():
            occ = label_occ.count(label_id) + 1
            new_filename = f"{label_id}_{occ}.jpg"
            coordinates = {
                "xmin": int(row.xmin),
                "ymin": int(row.ymin),
                "xmax": int(row.xmax),
                "ymax": int(row.ymax),
            }
            crop_picture(image_raw, path, new_filename, **coordinates)
            label_occ.append(label_id)

        crops_for_this_image = len(glob.glob(os.path.join(path, f"{label_id}_*.jpg")))
        total_crops += crops_for_this_image
        print(f"{filename} generated {crops_for_this_image} crops")

    print(f"\nTotal crops generated: {total_crops}")
    print(f"\nThe images have been successfully saved in {path}")
