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
        """
        if not os.path.exists(self.path_to_model):
            raise FileNotFoundError(f"Model file '{self.path_to_model}' not found.")
        if os.path.getsize(self.path_to_model) == 0:
            raise IOError(f"Model file '{self.path_to_model}' is empty.")
        try:
            print("Loading model from:", self.path_to_model)
            model = Model.load(self.path_to_model, self.classes)
            print("Model loaded successfully")
        except Exception as e:
            print("Model load failed:", e)
            raise
        return model
    
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