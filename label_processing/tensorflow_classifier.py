# Import third-party libraries
import numpy as np
import pandas as pd
import cv2
import glob, os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import warnings

# Import the necessary module from the 'label_processing' module package
from label_processing import utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


#--------------------------------Predict Classes--------------------------------#


def get_model(path_to_model: str) -> tf.keras.Sequential:
    """
    Load a trained Keras Sequential image classifier model.

    Args:
        path_to_model (str): Path to the model file.

    Returns:
        model (tf.keras.Sequential): Trained Keras Sequential image classifier model.
    """
    print("\nCalling classification model")
    model = tf.keras.models.load_model(path_to_model)
    return model

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