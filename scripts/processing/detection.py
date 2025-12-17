#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import sys
import time
import warnings
import pickle
import hashlib
from pathlib import Path
import pandas as pd
import torch

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Import project configuration
from label_processing.config import get_model_path, get_project_root

# Import the necessary module from the 'label_processing' module package
import label_processing.label_detection as scrop
from label_processing.label_detection import create_crops
from detecto.core import Model

# Constants
THRESHOLD = 0.8
PROCESSES = 12

class OptimizedPredictLabel:
    """
    Optimized version of PredictLabel with caching and streamlined loading.
    """
    
    def __init__(self, path_to_model: str, classes: list, 
                 threshold: float = 0.8, use_cache: bool = True):
        """
        Initialize with optimized model loading.
        
        Args:
            path_to_model (str): Path to the model file
            classes (list): List of class names
            threshold (float): Detection threshold
            use_cache (bool): Whether to use model caching
        """
        self.path_to_model = Path(path_to_model)
        self.classes = classes
        self.threshold = threshold
        self.use_cache = use_cache
        
        # Setup caching
        self.cache_dir = Path.home() / '.entomological_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load model with optimizations
        self.model = self.load_model_optimized()
    
    def _get_model_hash(self) -> str:
        """Generate hash of model file for cache validation."""
        with open(self.path_to_model, 'rb') as f:
            # Read first 1MB and last 1MB for quick hash
            first_chunk = f.read(1024 * 1024)
            f.seek(-1024 * 1024, 2)  # Seek to last MB
            last_chunk = f.read(1024 * 1024)
        
        combined = first_chunk + last_chunk + str(self.path_to_model.stat().st_mtime).encode()
        return hashlib.md5(combined).hexdigest()
    
    def _get_cache_path(self) -> Path:
        """Get path for cached model."""
        model_hash = self._get_model_hash()
        return self.cache_dir / f"model_{model_hash}.pkl"
    
    def load_model_optimized(self) -> Model:
        """
        Load model with optimized strategy.
        """
        print(f"Loading model from: {self.path_to_model}")
        start_time = time.perf_counter()
        
        # Try to load from cache first
        if self.use_cache:
            cached_model = self._try_load_from_cache()
            if cached_model is not None:
                load_time = time.perf_counter() - start_time
                print(f" Model loaded from cache in {load_time:.2f}s")
                return cached_model
        
        # If cache miss, load model efficiently
        model = self._load_model_direct()
        
        # Cache the model for future use
        if self.use_cache:
            self._cache_model(model)
        
        load_time = time.perf_counter() - start_time
        print(f" Model loaded in {load_time:.2f}s")
        return model
    
    def _try_load_from_cache(self) -> Model:
        """Try to load model from cache."""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return None
        
        try:
            print("Attempting to load model from cache...")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Reconstruct model from cached state dict
            model = Model(self.classes)
            
            # Handle different model attribute names
            if hasattr(model, 'model'):
                model.model.load_state_dict(cached_data['state_dict'], strict=False)
            elif hasattr(model, '_model'):
                model._model.load_state_dict(cached_data['state_dict'], strict=False)
            else:
                model.load_state_dict(cached_data['state_dict'], strict=False)
            
            return model
            
        except Exception as e:
            print(f"Cache loading failed: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None
    
    def _load_model_direct(self) -> Model:
        """Load model directly with optimized settings."""
        # Set up optimal environment
        self._setup_optimal_environment()
        
        try:
            # First try the most compatible approach for PyTorch 2.6+
            print("Loading with PyTorch 2.6+ compatibility...")
            
            # Monkey-patch torch.load temporarily
            original_torch_load = torch.load
            
            def optimized_load(*args, **kwargs):
                kwargs['weights_only'] = False
                kwargs['map_location'] = 'cpu'  # Always load to CPU first for compatibility
                return original_torch_load(*args, **kwargs)
            
            torch.load = optimized_load
            
            try:
                model = Model.load(str(self.path_to_model), self.classes)
                return model
            finally:
                torch.load = original_torch_load
                
        except Exception as e:
            print(f"Direct loading failed: {e}")
            # Fallback to manual loading
            return self._load_model_manual()
    
    def _load_model_manual(self) -> Model:
        """Manual model loading as fallback."""
        print("Using manual loading fallback...")
        
        # Load state dict manually
        state_dict = torch.load(str(self.path_to_model), 
                              map_location='cpu', 
                              weights_only=False)
        
        # Create new model and load state
        model = Model(self.classes)
        
        if isinstance(state_dict, dict):
            model.model.load_state_dict(state_dict, strict=False)
        else:
            # Handle model object
            if hasattr(state_dict, 'state_dict'):
                model.model.load_state_dict(state_dict.state_dict(), strict=False)
            else:
                raise Exception(f"Unknown model format: {type(state_dict)}")
        
        return model
    
    def _cache_model(self, model: Model):
        """Cache the loaded model for future use."""
        try:
            cache_path = self._get_cache_path()
            
            # Handle different model attribute names
            if hasattr(model, 'model'):
                state_dict = model.model.state_dict()
            elif hasattr(model, '_model'):
                state_dict = model._model.state_dict()
            else:
                state_dict = model.state_dict()
            
            cached_data = {
                'state_dict': state_dict,
                'classes': self.classes,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f" Model cached for future use")
            
        except Exception as e:
            print(f"Warning: Could not cache model: {e}")
    
    def _setup_optimal_environment(self):
        """Setup optimal environment for loading."""
        # Disable CUDA for loading (can enable later for inference)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Optimize thread usage
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        
        # Set environment variables for stability
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    
    def class_prediction(self, jpg_path: Path) -> pd.DataFrame:
        """
        Predict labels for a given JPG file.
        
        Args:
        jpg_path (Path): Path to the JPG file
            
        Returns:
            pd.DataFrame: Prediction results
        """
        import detecto.utils
        
        image = detecto.utils.read_image(str(jpg_path))
        predictions = self.model.predict(image)
        labels, boxes, scores = predictions
        
        entries = []
        for i, labelname in enumerate(labels):
            entry = {
                'filename': jpg_path.name,
                'class': labelname,
                'score': scores[i].item(),
                'xmin': boxes[i][0],
                'ymin': boxes[i][1],
                'xmax': boxes[i][2],
                'ymax': boxes[i][3]
            }
            entries.append(entry)
        
        return pd.DataFrame(entries)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute label detection on entomological specimen images with performance optimizations."
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-j', '--input-dir',
        type=str,
        help='Directory containing specimen images'
    )
    input_group.add_argument(
        '-i', '--input-image',
        type=str,
        help='Single image file to process'
    )
    
    # Output directory (required)
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory where results will be saved'
    )
    
    # Optional parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.8,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of images processed simultaneously (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for processing (default: auto)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable model caching'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear model cache before running'
    )

    return parser.parse_args()


def clear_model_cache():
    """Clear all cached models."""
    cache_dir = Path.home() / '.entomological_cache'
    if cache_dir.exists():
        for cache_file in cache_dir.glob('model_*.pkl'):
            try:
                cache_file.unlink()
                print(f"Removed cache file: {cache_file.name}")
            except Exception as e:
                print(f"Could not remove {cache_file.name}: {e}")
        print("Model cache cleared.")
    else:
        print("No cache directory found.")


def setup_device(device_arg: str) -> str:
    """
    Setup optimal device for inference.
    
    Args:
        device_arg: Device argument from command line
        
    Returns:
        str: Best available device
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f" Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print(" Using Apple Metal Performance Shaders (MPS)")
        else:
            device = 'cpu'
            print(f" Using CPU with {torch.get_num_threads()} threads")
    else:
        device = device_arg
        print(f" Using specified device: {device}")
    
    return device


def main():
    """
    Main execution function with performance optimizations.
    """
    start_time = time.perf_counter()
    args = parse_arguments()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_model_cache()
        return
    
    # Use centralized configuration for model path
    try:
        MODEL_PATH = get_model_path("detection")
    except Exception as e:
        print(f"Error getting model path: {e}")
        print("Please ensure the model file exists or set the ENTOMOLOGICAL_DETECTION_MODEL_PATH environment variable.")
        return
    
    # Handle input (directory or single file)
    if args.input_dir:
        jpg_dir = Path(args.input_dir)
        input_type = "directory"
    else:
        # Single file input
        single_file = Path(args.input_image)
        if not single_file.exists():
            print(f"Error: Input file '{single_file}' does not exist.")
            return
        jpg_dir = single_file.parent
        input_type = "single_file"
        print(f"Processing single file: {single_file.name}")
    
    out_dir = args.output_dir
    confidence_threshold = args.confidence
    batch_size = args.batch_size
    use_cache = not args.no_cache
    classes = ["label"]
    
    # Validate paths
    if not os.path.exists(out_dir):
        print(f"Creating output directory: {out_dir}")
        os.makedirs(out_dir)
    if not MODEL_PATH.exists():
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    if input_type == "directory" and not jpg_dir.exists():
        print(f"Error: Input directory '{jpg_dir}' does not exist.")
        return
    
    print(f"Using confidence threshold: {confidence_threshold}")
    print(f"Using batch size: {batch_size}")
    print(f"Model caching: {'enabled' if use_cache else 'disabled'}")
    
    # Setup device
    device = setup_device(args.device)
    
    try:
        # Initialize optimized predictor
        predictor = OptimizedPredictLabel(MODEL_PATH, classes, use_cache=use_cache)
        
        # Move model to selected device if not CPU
        if device != 'cpu':
            try:
                if hasattr(predictor.model, 'model'):
                    predictor.model.model = predictor.model.model.to(device)
                elif hasattr(predictor.model, '_model'):
                    predictor.model._model = predictor.model._model.to(device)
                else:
                    # Try to move the predictor model directly
                    predictor.model = predictor.model.to(device)
                print(f" Model moved to {device}")
            except Exception as e:
                print(f"Warning: Could not move model to {device}, using CPU: {e}")
        
        model_load_time = time.perf_counter() - start_time
        print(f" Total model setup time: {model_load_time:.2f}s")
        
        # Prediction phase
        prediction_start = time.perf_counter()
        
        if input_type == "single_file":
            print(f"Processing single file: {single_file}")
            df = predictor.class_prediction(single_file)
            if df.empty:
                df = pd.DataFrame(columns=['filename', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
        else:
            # For CPU inference, use sequential processing to avoid multiprocessing overhead
            # For GPU/MPS, parallel processing could be beneficial
            if device == 'cpu':
                print("Processing images sequentially (CPU mode)...")
                # Collect image files
                file_names = [p for p in sorted(jpg_dir.iterdir()) if scrop.is_image_file(p)]
                print(f"Found {len(file_names)} images to process")
                
                # Process sequentially
                results = []
                for i, file_path in enumerate(file_names, 1):
                    print(f"Processing {i}/{len(file_names)}: {file_path.name}", end='\r')
                    result_df = predictor.class_prediction(file_path)
                    if not result_df.empty:
                        results.append(result_df)
                
                print()  # New line after progress
                df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
            else:
                # For GPU/MPS, use parallel processing
                processes = min(PROCESSES, batch_size) if batch_size < PROCESSES else PROCESSES
                df = scrop.prediction_parallel(jpg_dir, predictor, processes)
        
        prediction_time = time.perf_counter() - prediction_start
        print(f" Prediction completed in {prediction_time:.2f}s")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    if df.empty:
        print("No valid predictions were generated. Skipping further processing.")
        return
    
    try:
        df = scrop.clean_predictions(jpg_dir, df, confidence_threshold, out_dir=out_dir)
    except Exception as e:
        print(f"Error cleaning predictions: {e}")
        return
    
    detection_total_time = time.perf_counter() - start_time
    print(f"Detection finished in {detection_total_time:.2f}s")
    
    try:
        crop_start = time.perf_counter()
        create_crops(jpg_dir, df, out_dir=out_dir)
        crop_time = time.perf_counter() - crop_start
        print(f" Cropping completed in {crop_time:.2f}s")
    except Exception as e:
        print(f"Error during cropping: {e}")
        return
    
    total_time = time.perf_counter() - start_time
    print(f"\n" + "="*50)
    print(f" PROCESSING COMPLETED")
    print(f" Total time: {total_time:.2f}s")
    print(f"  - Model loading: {model_load_time:.2f}s ({model_load_time/total_time*100:.1f}%)")
    print(f"  - Prediction: {prediction_time:.2f}s ({prediction_time/total_time*100:.1f}%)")
    print(f"  - Cropping: {crop_time:.2f}s ({crop_time/total_time*100:.1f}%)")
    print(f" Results saved to: {out_dir}")
    print(f"  - CSV file: {out_dir}/input_predictions.csv")
    print(f"  - Cropped images: {out_dir}/input_cropped/")
    print("="*50)


if __name__ == '__main__':
    main()

