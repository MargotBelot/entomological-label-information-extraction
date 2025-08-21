# Import third-party libraries
import numpy as np
import pandas as pd
import cv2
import glob, os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import warnings
import platform
import sys

# Import the necessary module from the 'label_processing' module package
from label_processing import utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


#--------------------------------Predict Classes--------------------------------#


def get_model(path_to_model: str) -> tf.keras.Sequential:
    """
    Load a trained Keras Sequential image classifier model with cross-platform compatibility.

    Args:
        path_to_model (str): Path to the model file.

    Returns:
        model (tf.keras.Sequential): Trained Keras Sequential image classifier model.
    """
    print("\nCalling classification model")
    
    # Set up cross-platform environment
    _setup_tensorflow_cross_platform_environment()
    
    # Try multiple loading strategies for cross-platform compatibility
    loading_strategies = [
        # Strategy 1: Standard TensorFlow loading
        lambda: tf.keras.models.load_model(path_to_model),
        # Strategy 2: Loading with compile=False to avoid optimizer issues
        lambda: tf.keras.models.load_model(path_to_model, compile=False),
        # Strategy 3: Loading with custom options for protobuf compatibility
        lambda: _load_with_protobuf_compatibility(path_to_model),
        # Strategy 4: Loading with SavedModel format explicitly
        lambda: _load_with_saved_model_format(path_to_model),
    ]
    
    last_error = None
    for i, strategy in enumerate(loading_strategies, 1):
        try:
            print(f"Trying TensorFlow loading strategy {i}...")
            model = strategy()
            print("TensorFlow model loaded successfully")
            return model
        except Exception as e:
            print(f"TensorFlow strategy {i} failed: {e}")
            last_error = e
            continue
    
    # If all strategies fail, raise the last error with helpful message
    print(f"All TensorFlow loading strategies failed. Last error: {last_error}")
    raise Exception(f"Failed to load TensorFlow model from {path_to_model}. "
                   f"This might be due to protobuf version incompatibility or "
                   f"model corruption. Last error: {last_error}")

def class_prediction(model: tf.keras.Sequential, class_names: list, jpg_dir: str, out_dir=None) -> pd.DataFrame:
    """
    Create a dataframe with predicted classes for each picture.

    Args:
        model (tf.keras.Sequential): Trained Keras Sequential image classifier model.
        class_names (list): Model's predicted classes.
        jpg_dir (str): Path to the directory containing the original jpgs.
        out_dir (str): Path where the CSV file will be stored.

    Returns:
        DataFrame (pd.DataFrame): Pandas DataFrame with the predicted results.
    """
    utils.check_dir(jpg_dir)
    print("\nPredicting classes")
    all_predictions = []
    img_width = 180
    img_height = 180
    for file in glob.glob(f"{jpg_dir}/*.jpg"):
        image = tf.keras.utils.load_img(file, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        entry = {}
        entry['filename'] = os.path.basename(file)
        entry['class'] = class_names[np.argmax(score)]
        entry['score'] = 100 * np.max(score)
        all_predictions.append(entry)
    df = pd.DataFrame(all_predictions)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.realpath(jpg_dir))
    filename = f"{Path(jpg_dir).stem}_prediction_classifer.csv"
    csv_path = f"{out_dir}/{filename}"
    df.to_csv(csv_path)
    print(f"\nThe CSV file {filename} has been successfully saved in {out_dir}")
    return df


#--------------------------------Save Pictures--------------------------------#


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

def rename_picture(img_raw: np.ndarray , path: str, filename: str, pic_class: str) -> None:
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

def filter_pictures(jpg_dir: Path, dataframe: pd.DataFrame, out_dir: Path = Path(os.getcwd())) -> None:
    """
    Create new folders for each class of the newly named classified pictures.

    Args:
        jpg_dir (str): Path to directory with jpgs.
        dataframe (pd.DataFrame): Pandas DataFrame with class predictions.
        out_dir (Path): Path to the target directory to save the cropped jpgs.
    """
    create_dirs(dataframe, out_dir)  # Create directories for every class

    for filepath in glob.glob(os.path.join(jpg_dir, '*.jpg')):
        filename = os.path.basename(filepath)
        match = dataframe[dataframe.filename == filename]
        image_raw = cv2.imread(filepath)
        label_id = Path(filename).stem
        for _, row in match.iterrows():
            pic_class = row['class']
            filename = make_file_name(label_id, pic_class)
            rename_picture(image_raw, out_dir, filename, pic_class)
    print(f"\nThe images have been successfully saved in {out_dir}")


#--------------------------------Cross-Platform Compatibility--------------------------------#


def _setup_tensorflow_cross_platform_environment():
    """Setup TensorFlow environment for cross-platform compatibility."""
    # Force CPU-only execution to avoid CUDA/GPU issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
    
    # Linux-specific optimizations
    if platform.system() == 'Linux':
        # Disable problematic optimizations on Linux
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Configure TensorFlow for better Linux compatibility
        tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
        
        # Set memory growth for any GPU that might be detected
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
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
            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
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
        if hasattr(imported, 'signatures'):
            # Try to get the serving signature
            if 'serving_default' in imported.signatures:
                # Wrap in a Keras model-like interface
                signature = imported.signatures['serving_default']
                
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
