# Import third-party libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

def metrics(target: list, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path = Path(os.getcwd())) -> str:
    """
    Build a text report showing the main classification metrics,
    to measure the quality of predictions of the classification model, and save it to a text file.

    Args:
        target (list): Names matching the classes.
        pred (pd.DataFrame): Predicted classes.
        gt (pd.DataFrame): Ground truth classes.
        out_dir (Path): Directory where the report file will be saved.

    Returns:
        str: Classification report as a text output.
    """
    try:
        report_file = out_dir / "classification_report.txt"
        accuracy = accuracy_score(gt, pred) * 100
        report = classification_report(gt, pred, target_names=target)
        
        print("Accuracy Score ->", accuracy)
        print(report)
        
        with open(report_file, 'w') as file:
            file.write(f"Accuracy Score -> {accuracy}\n")
            file.write(report)
        
        print(f"\nThe Classification Report has been successfully saved in {report_file}")
        return report
    except Exception as e:
        print(f"Error computing classification metrics: {e}")
        return ""

def cm(target: list, pred: pd.DataFrame, gt: pd.DataFrame, out_dir: Path = Path(os.getcwd())) -> None:
    """
    Compute confusion matrix to evaluate the performance of the classification.

    Args:
        target (list): Names matching the classes.
        pred (pd.DataFrame): Predicted classes.
        gt (pd.DataFrame): Ground truth classes.
        out_dir (Path): Path to the target directory to save the confusion matrix plot.
    """
    try:
        cm = confusion_matrix(gt, pred)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(15,10))
        matrix = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target, yticklabels=target, cmap="Greens",
                              annot_kws={"size": 14})
        plt.ylabel('Ground truth', fontsize=18)
        plt.xlabel('Predictions', labelpad=30, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        figure = matrix.get_figure()
        filename = f"{out_dir.stem}_cm.png"
        cm_path = out_dir / filename
        figure.savefig(cm_path)
        plt.close(fig)
        
        print(f"\nThe Confusion Matrix has been successfully saved in {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
