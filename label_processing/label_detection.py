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
from detecto.core import Model
import pickle

import warnings
import platform
import sys

# Suppress torchvision deprecation warnings from detecto library
warnings.filterwarnings('ignore', message='The parameter \'pretrained\' is deprecated.*')
warnings.filterwarnings('ignore', message='Arguments other than a weight enum.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


#---------------------Image Segmentation---------------------#


class PredictLabel():
    """
    Class for predicting labels using a trained object detection model.

    Attributes:
        path_to_model (str): Path to the trained model file.
        classes (list): List of classes used in the model.
        jpg_path (str|Path|None): Path to a specific JPG file for prediction.
        threshold (float): Threshold value for scores. Defaults to 0.8.
        model (detecto.core.Model): Trained object detection model.
    """

    def __init__(self, path_to_model: str, classes: list,
                 jpg_path: Union[str, Path, None] = None,
                 threshold: float = 0.8) -> None:
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
        Includes cross-platform compatibility fixes.
        """
        if not os.path.exists(self.path_to_model):
            raise FileNotFoundError(f"Model file '{self.path_to_model}' not found.")
        if os.path.getsize(self.path_to_model) == 0:
            raise IOError(f"Model file '{self.path_to_model}' is empty.")
        
        print("Loading model from:", self.path_to_model)
        
        # Set environment for cross-platform compatibility
        self._setup_cross_platform_environment()
        
        # Try multiple loading strategies for cross-platform compatibility
        loading_strategies = [
            # Strategy 1: PyTorch 2.6+ compatible loading with weights_only=False
            lambda: self._load_pytorch_2_6_compatible(),
            # Strategy 2: Direct detecto loading (legacy)
            lambda: Model.load(self.path_to_model, self.classes),
            # Strategy 3: Force CPU loading (for CUDA/CPU mismatch issues)
            lambda: self._load_with_cpu_fallback(),
            # Strategy 4: Force weights_only=False (for corrupted pickle issues)
            lambda: self._load_with_weights_only_false(),
            # Strategy 5: Pickle protocol fallback
            lambda: self._load_with_pickle_protocol(),
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
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Set multiprocessing start method for Linux compatibility
        if platform.system() == 'Linux':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # Method already set, ignore
                pass
        
        # Set PyTorch thread limits for stable performance
        torch.set_num_threads(1)
        
        # Disable MKL optimizations that can cause issues on some Linux distributions
        if platform.system() == 'Linux':
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    def _load_pytorch_2_6_compatible(self):
        """PyTorch 2.6+ compatible loading with explicit weights_only=False."""
        try:
            # PyTorch 2.6+ requires explicit weights_only=False for models with custom objects
            # This is the most direct fix for the "Unsupported operand 118" error
            print("Attempting PyTorch 2.6+ compatible loading with weights_only=False")
            
            # Monkey-patch torch.load to always use weights_only=False
            original_torch_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Temporarily replace torch.load
            torch.load = patched_load
            
            try:
                # Use detecto's built-in loading with patched torch.load
                model = Model.load(self.path_to_model, self.classes)
                return model
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
                
        except Exception as e:
            print(f"PyTorch 2.6 compatible loading failed: {e}")
            # Fallback to manual loading with weights_only=False
            return self._load_with_manual_weights_only_false()
    
    def _load_with_manual_weights_only_false(self):
        """Manual loading with weights_only=False and proper error handling."""
        try:
            print("Attempting manual loading with weights_only=False")
            # Force weights_only=False with explicit error handling for corrupted files
            state_dict = torch.load(self.path_to_model, map_location='cpu', weights_only=False)
            
            # Create a new model instance
            model = Model(self.classes)
            
            # Handle different possible state dict formats
            if isinstance(state_dict, dict):
                # If it's a plain state dict
                try:
                    model.model.load_state_dict(state_dict, strict=False)
                except Exception:
                    # Try alternative model attribute
                    model._model.load_state_dict(state_dict, strict=False)
            else:
                # If it's a model object, extract state dict
                if hasattr(state_dict, 'state_dict'):
                    actual_state_dict = state_dict.state_dict()
                    try:
                        model.model.load_state_dict(actual_state_dict, strict=False)
                    except Exception:
                        model._model.load_state_dict(actual_state_dict, strict=False)
                else:
                    raise Exception(f"Unknown model format: {type(state_dict)}")
            
            print("Manual loading successful")
            return model
            
        except Exception as e:
            print(f"Manual weights_only=False loading failed: {e}")
            raise e
    
    def _load_with_cpu_fallback(self):
        """Load model with CPU map_location to handle CUDA/CPU mismatches."""
        try:
            # First try with weights_only=False to handle PyTorch 2.6+ compatibility
            state_dict = torch.load(self.path_to_model, map_location='cpu', weights_only=False)
            # Create a new model instance and load the state dict
            model = Model(self.classes)
            # Handle potential state dict key mismatches
            if hasattr(model.model, 'load_state_dict'):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                # Alternative loading for different detecto versions
                model._model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            print(f"CPU fallback failed: {e}")
            # Try even more basic loading
            return self._load_with_basic_torch()
    
    def _load_with_weights_only(self):
        """Load model using weights_only=True for newer PyTorch versions."""
        try:
            # Try with weights_only=True (PyTorch 1.13+)
            state_dict = torch.load(self.path_to_model, map_location='cpu', weights_only=True)
            model = Model(self.classes)
            model.model.load_state_dict(state_dict)
            return model
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            return self._load_with_cpu_fallback()
    
    def _load_with_weights_only_false(self):
        """Load model with weights_only=False to handle pickle issues."""
        try:
            # Force weights_only=False for corrupted pickle files
            state_dict = torch.load(self.path_to_model, map_location='cpu', weights_only=False)
            model = Model(self.classes)
            model.model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            print(f"weights_only=False loading failed: {e}")
            return self._load_with_basic_torch()
    
    def _load_with_pickle_protocol(self):
        """Load model with different pickle protocols to handle version mismatches."""
        try:
            # Try loading with pickle protocol compatibility
            with open(self.path_to_model, 'rb') as f:
                # Load raw pickle data
                state_dict = pickle.load(f)
                
            model = Model(self.classes)
            if isinstance(state_dict, dict):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                # If it's a full model object, extract the state dict
                if hasattr(state_dict, 'state_dict'):
                    model.model.load_state_dict(state_dict.state_dict(), strict=False)
                else:
                    raise Exception("Unknown state dict format")
            return model
        except Exception as e:
            print(f"Pickle protocol loading failed: {e}")
            return self._load_with_basic_torch()
    
    def _load_with_basic_torch(self):
        """Most basic torch loading as last resort."""
        # Just load the state dict and create a minimal wrapper
        # Use weights_only=False for PyTorch 2.6+ compatibility
        state_dict = torch.load(self.path_to_model, map_location=torch.device('cpu'), weights_only=False)
        
        # Try to create a detecto model the most basic way possible
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from detecto.core import Model
            
            # Create base model
            model = Model(self.classes)
            
            # Force load state dict with any necessary key adjustments
            try:
                model.model.load_state_dict(state_dict, strict=False)
            except:
                # If that fails, try alternative attribute names
                if hasattr(model, '_model'):
                    model._model.load_state_dict(state_dict, strict=False)
                else:
                    raise Exception("Could not find model attribute to load state dict")
            
            return model
            
        except Exception as e:
            raise Exception(f"All loading methods failed. Last attempt error: {e}")
    
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
        image = detecto.utils.read_image(str(jpg_path))
        predictions = self.model.predict(image)
        labels, boxes, scores = predictions
        
        entries = []
        for i, labelname in enumerate(labels):
            entry = {}
            entry['filename'] = jpg_path.name
            entry['class'] = labelname
            entry['score'] = scores[i].item()
            entry['xmin'] = boxes[i][0]
            entry['ymin'] = boxes[i][1]
            entry['xmax'] = boxes[i][2]
            entry['ymax'] = boxes[i][3]
            entries.append(entry)
        return pd.DataFrame(entries)


def prediction_parallel(jpg_dir: Union[str, Path], predictor: PredictLabel,
                        n_processes: int) -> pd.DataFrame:
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

    file_names: list[Path] = list(jpg_dir.glob("*.jpg"))

    # Validate each image before processing
    valid_files = []
    for file in file_names:
        image = cv2.imread(str(file))
        if image is None:
            print(f"Skipping corrupted image: {file}")
        else:
            valid_files.append(file)

    mp.set_start_method('spawn', force=True)
    with mp.Pool(n_processes) as executor:
        results = executor.map(predictor.class_prediction, valid_files)

    final_results = []
    map(final_results.extend, results)
    return pd.concat(results, ignore_index=True)

def clean_predictions(jpg_dir: Path, dataframe: pd.DataFrame,
                      threshold: float, out_dir=None) -> pd.DataFrame:
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
    colnames = ['score', 'xmin', 'ymin', 'xmax', 'ymax']
    for header in colnames:
        dataframe[header] = dataframe[header].astype('str').str.\
            extractall('(\d+.\d+)').unstack().fillna('').sum(axis=1).astype(float)
    dataframe = dataframe.loc[dataframe['score'] >= threshold]
    dataframe[['xmin', 'ymin','xmax','ymax']] = \
        dataframe[['xmin', 'ymin','xmax','ymax']].fillna('0')
    
    if out_dir is None:
        parent_dir = jpg_dir.resolve().parent
    else:
        parent_dir = out_dir
    filename = f"{jpg_dir.stem}_predictions.csv"
    csv_path = f"{parent_dir}/{filename}"
    dataframe.to_csv(csv_path)
    print(f"\nThe csv_file {filename} has been successfully saved in {out_dir}")
    return dataframe


#---------------------Image Cropping---------------------#    


def crop_picture(img_raw: np.ndarray, path: str,
                 filename: str, **coordinates) -> None:
    """
    Crop the picture using the given coordinates.

    Args:
        img_raw (numpy.ndarray): Input JPG converted to a numpy matrix by cv2.
        path (str): Path where the picture should be saved.
        filename (str): Name of the picture.
        coordinates: Coordinates for cropping.
    """
    xmin = coordinates['xmin']
    ymin = coordinates['ymin']
    xmax = coordinates['xmax']
    ymax = coordinates['ymax']
    filepath = f"{path}/{filename}"
    crop = img_raw[ymin:ymax, xmin:xmax]
    cv2.imwrite(filepath, crop)


def create_crops(jpg_dir: Path, dataframe: pd.DataFrame,
                 out_dir: Path = Path(os.getcwd())) -> None:
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
    for filepath in glob.glob(os.path.join(dir_path, '*.jpg')):
        if not os.path.exists(filepath):
            print(f"File cannot be found: {filepath}")
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
            coordinates = {'xmin': int(row.xmin), 'ymin': int(row.ymin),
                           'xmax': int(row.xmax), 'ymax': int(row.ymax)}
            crop_picture(image_raw, path, new_filename, **coordinates)
            label_occ.append(label_id)

        crops_for_this_image = len(glob.glob(os.path.join(path, f"{label_id}_*.jpg")))
        total_crops += crops_for_this_image
        print(f"{filename} generated {crops_for_this_image} crops")
    
    print(f"\nTotal crops generated: {total_crops}")
    print(f"\nThe images have been successfully saved in {path}")