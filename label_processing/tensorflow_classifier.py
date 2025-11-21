# Import third-party libraries
import glob
import os
import platform
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import check_dir, glob_image_files, validate_image_integrity

# Suppress warning messages during execution
warnings.filterwarnings("ignore")


# --------------------------------Predict Classes--------------------------------#

def class_prediction(
    model: tf.keras.Sequential,
    class_names: list,
    jpg_dir: str,
    out_dir=None,
    batch_size: int = 32,
    max_images: int = 10000,
) -> pd.DataFrame:
    """
    Create a dataframe with predicted classes for each picture with memory-safe batch processing.

    Args:
        model (tf.keras.Sequential): Trained Keras Sequential image classifier model.
        class_names (list): Model's predicted classes.
        jpg_dir (str): Path to the directory containing the original jpgs.
        out_dir (str): Path where the CSV file will be stored.
        batch_size (int): Number of images to process in each batch (default: 32)
        max_images (int): Maximum number of images to process (default: 10000)

    Returns:
        DataFrame (pd.DataFrame): Pandas DataFrame with the predicted results.
    """
    check_dir(jpg_dir)
    print("\nPredicting classes with memory-safe batch processing")

    # Get all image files
    image_files = glob_image_files(jpg_dir, check_integrity=True, raise_when_no_entries=True)

    # SECURITY: Limit total number of images to prevent resource exhaustion
    if len(image_files) > max_images:
        print(
            f"SECURITY WARNING: Too many images ({len(image_files)} > {max_images}). Processing only first {max_images}."
        )
        image_files = image_files[:max_images]

    all_predictions = []
    img_width = 180
    img_height = 180

    # Process images in batches to prevent memory exhaustion
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch_files)} images)"
        )

        # Validate all images in batch first
        valid_batch_files = []
        for file in batch_files:
            if validate_image_integrity(
                file, max_size_mb=10, max_dimensions=(4000, 4000)
            ):
                valid_batch_files.append(file)
            else:
                print(f"SECURITY WARNING: Skipping unsafe image: {file}")

        # Process valid images in current batch
        for file in valid_batch_files:
            try:
                # SECURITY: Use safe image loading with error handling
                image = tf.keras.utils.load_img(
                    file, target_size=(img_height, img_width)
                )
                img_array = tf.keras.utils.img_to_array(image)
                img_array = tf.expand_dims(img_array, 0)

                # SECURITY: Clear GPU memory after each prediction to prevent accumulation
                # Note: verbose parameter removed for SavedModel compatibility
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                entry = {}
                entry["filename"] = os.path.basename(file)
                entry["class"] = class_names[np.argmax(score)]
                entry["score"] = 100 * np.max(score)
                all_predictions.append(entry)

                # SECURITY: Clear variables to free memory
                del img_array, predictions, score, image

            except Exception as e:
                print(f"SECURITY ERROR: Failed to process image {file}: {e}")
                continue

        # SECURITY: Force garbage collection after each batch
        import gc

        gc.collect()
    df = pd.DataFrame(all_predictions)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.realpath(jpg_dir))
    filename = f"{Path(jpg_dir).stem}_prediction_classifer.csv"
    csv_path = f"{out_dir}/{filename}"
    df.to_csv(csv_path)
    print(f"\nThe CSV file {filename} has been successfully saved in {out_dir}")
    return df


# --------------------------------Save Pictures--------------------------------#


def create_dirs(dataframe: pd.DataFrame, path: str) -> None:
    """
    Create separate directories for every class.

    Args:
        dataframe (pd.Dataframe): DataFrame containing the classes as a column.
        path (str): Path of the chosen directory.
    """
    uniques = dataframe["class"].unique()
    for uni_class in uniques:
        Path(f"{path}/{uni_class}").mkdir(parents=True, exist_ok=True)


def make_file_name(label_id: str, pic_class: str) -> None:
    """
    Create a fitting filename.

    Args:
        label_id (str): String containing the label id.
        pic_class (str): Class of the label.

    Returns:
        filename (str): The created filename.
    """
    filename = f"{label_id}_{pic_class}.jpg"
    return filename


def rename_picture(
    img_raw: np.ndarray, path: str, filename: str, pic_class: str
) -> None:
    """
    Rename the pictures using the predicted class.

    Args:
        img_raw (numpy.ndarray): Input jpg converted to a numpy matrix by cv2.
        path (str): Path where the picture should be saved.
        filename (str): Name of the picture.
        pic_class (str): Class of the label.
    """
    filepath = f"{path}/{pic_class}/{filename}"
    cv2.imwrite(filepath, img_raw)


def filter_pictures(
    jpg_dir: Path, dataframe: pd.DataFrame, out_dir: Path = Path(os.getcwd())
) -> None:
    """
    Create new folders for each class of the newly named classified pictures.

    Args:
        jpg_dir (str): Path to directory with jpgs.
        dataframe (pd.DataFrame): Pandas DataFrame with class predictions.
        out_dir (Path): Path to the target directory to save the cropped jpgs.
    """
    # Check if dataframe has required columns
    if dataframe.empty:
        print("Warning: No predictions available for image filtering")
        return
        
    if 'class' not in dataframe.columns:
        print(f"Error: DataFrame missing 'class' column. Available columns: {list(dataframe.columns)}")
        return
        
    if 'filename' not in dataframe.columns:
        print(f"Error: DataFrame missing 'filename' column. Available columns: {list(dataframe.columns)}")
        return
    
    try:
        create_dirs(dataframe, out_dir)  # Create directories for every class
    except Exception as e:
        print(f"Error creating directories: {e}")
        return

    for filepath in glob.glob(os.path.join(jpg_dir, "*.jpg")):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        
        if match.empty:
            print(f"Warning: No prediction found for image {filename}")
            continue
            
        try:
            image_raw = cv2.imread(filepath)
            if image_raw is None:
                print(f"Warning: Could not load image {filepath}")
                continue
                
            label_id = Path(filename).stem
            for _, row in match.iterrows():
                if 'class' not in row or pd.isna(row['class']):
                    print(f"Warning: No valid class prediction for {filename}")
                    continue
                    
                pic_class = row["class"]
                filename = make_file_name(label_id, pic_class)
                rename_picture(image_raw, out_dir, filename, pic_class)
        except Exception as e:
            print(f"Error processing image {filepath}: {e}")
            continue
            
    print(f"\nThe images have been successfully saved in {out_dir}")


# ---------------------TensorFlow Model Loading cross-platform with Fallbacks---------------------#


def _setup_tensorflow_cross_platform_environment():
    """Setup TensorFlow environment for cross-platform compatibility."""
    # Force CPU-only execution to avoid CUDA/GPU issues
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

    # Linux-specific optimizations
    if platform.system() == "Linux":
        # Disable problematic optimizations on Linux
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # Configure TensorFlow for better Linux compatibility
        tf.config.set_visible_devices([], "GPU")  # Force CPU usage

        # Set memory growth for any GPU that might be detected
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass  # Ignore if GPU configuration fails


def _load_with_protobuf_compatibility(path_to_model: str) -> tf.keras.Sequential:
    """Load model with protobuf compatibility fixes for Linux.
    Arg:
        path_to_model (str): Path to the model file.
    Returns:
        model (tf.keras.Sequential): Loaded Keras Sequential model.
    """
    try:
        # Try to load with specific protobuf handling
        import google.protobuf.message

        # Set protobuf message size limits for large models
        google.protobuf.message.Message.SetAllowOversizeProtos(True)

        # Load with explicit options
        model = tf.keras.models.load_model(
            path_to_model,
            compile=False,
            custom_objects=None,
            options=tf.saved_model.LoadOptions(experimental_io_device="/job:localhost"),
        )

        return model
    except Exception as e:
        raise Exception(f"Protobuf compatibility loading failed: {e}")


def _load_with_saved_model_format(path_to_model: str) -> tf.keras.Sequential:
    """Load model using explicit SavedModel format.
    Args:
        path_to_model (str): Path to the model file.
    Returns:
        model (tf.keras.Sequential): Loaded Keras Sequential model or a wrapper."""
    try:
        # Load using tf.saved_model API directly
        imported = tf.saved_model.load(path_to_model)

        # Convert to Keras model if possible
        if hasattr(imported, "signatures"):
            # Try to get the serving signature
            if "serving_default" in imported.signatures:
                # Wrap in a Keras model-like interface
                signature = imported.signatures["serving_default"]

                class SavedModelWrapper:
                    def __init__(self, signature_fn):
                        self.signature_fn = signature_fn

                    def predict(self, x):
                        # Convert numpy array to tensor if needed
                        if isinstance(x, np.ndarray):
                            x = tf.convert_to_tensor(x, dtype=tf.float32)

                        # Call the signature function
                        result = self.signature_fn(x)

                        # Return numpy array for compatibility
                        if isinstance(result, dict):
                            # Get the first output if multiple outputs
                            output_key = list(result.keys())[0]
                            return result[output_key].numpy()
                        else:
                            return result.numpy()

                return SavedModelWrapper(signature)

        # If no serving signature, try to use the model directly
        return imported

    except Exception as e:
        raise Exception(f"SavedModel format loading failed: {e}")



def load_keras_model_with_fallbacks(model_path: str):
    """
    Load a Keras model using multiple fallback strategies for cross-platform compatibility.
    
    This function tries multiple loading strategies to handle compatibility issues
    between different Keras/TensorFlow versions and platforms.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        tf.keras.Model: Loaded Keras model.
        
    Raises:
        Exception: If all loading strategies fail.
    """    
    _setup_tensorflow_cross_platform_environment()
    
    loading_strategies = [
        lambda: tf.keras.models.load_model(model_path),
        lambda: tf.keras.models.load_model(model_path, compile=False),
        lambda: _load_with_protobuf_compatibility(model_path),
        lambda: _load_with_saved_model_format(model_path),
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

    raise Exception(
        f"Failed to load Keras model from {model_path}. "
        f"This might be due to protobuf version incompatibility or "
        f"model corruption. Last error: {last_error}"
    )
