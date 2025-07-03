# Import third-party libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_iou(pred_coords: tuple[float, float, float, float], 
                  gt_coords: tuple[str, float, float, float, float]) -> float:
    """
    Calculates Intersection over Union (IOU) scores by comparing predicted and ground truth segmentation coordinates.

    Args:
        pred_coords (tuple): Coordinates for the predicted bounding box (xmin, ymin, xmax, ymax).
        gt_coords (tuple): Coordinates for the ground truth bounding box (class, xmin, ymin, xmax, ymax).

    Returns:
        float: IOU score.
    """
    try:
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = pred_coords
        _, xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_coords
        
        x0_I = max(xmin_pred, xmin_gt)
        y0_I = max(ymin_pred, ymin_gt)
        x1_I = min(xmax_pred, xmax_gt)
        y1_I = min(ymax_pred, ymax_gt)
        
        width_I = max(0, x1_I - x0_I)
        height_I = max(0, y1_I - y0_I)
        intersection = width_I * height_I
        
        width_A, height_A = xmax_pred - xmin_pred, ymax_pred - ymin_pred
        width_B, height_B = xmax_gt - xmin_gt, ymax_gt - ymin_gt
        union = (width_A * height_A) + (width_B * height_B) - intersection
        
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print(f"Error calculating IOU: {e}")
        return 0.0

def comparison(df_pred_filename: pd.DataFrame, df_gt_filename: pd.DataFrame) -> pd.DataFrame:
    """
    Compare bounding box coordinates and calculate IOU scores.

    Args:
        df_pred_filename (pd.DataFrame): DataFrame with predicted labels.
        df_gt_filename (pd.DataFrame): DataFrame with ground truth labels.

    Returns:
        pd.DataFrame: DataFrame with added IOU scores.
    """
    try:
        max_scores, max_coords = [], {key: [] for key in ['class_gt', 'xmin_gt', 'ymin_gt', 'xmax_gt', 'ymax_gt']}
        
        for _, row_pred in df_pred_filename.iterrows():
            max_score, best_match = 0, {}
            pred_coords = (row_pred["xmin_pred"], row_pred["ymin_pred"], row_pred["xmax_pred"], row_pred["ymax_pred"])
            
            for _, row_gt in df_gt_filename.iterrows():
                gt_coords = (row_gt["class_gt"], row_gt["xmin_gt"], row_gt["ymin_gt"], row_gt["xmax_gt"], row_gt["ymax_gt"])
                iou = calculate_iou(pred_coords, gt_coords)
                if iou > max_score:
                    max_score, best_match = iou, row_gt
            
            max_scores.append(max_score)
            for key in max_coords:
                max_coords[key].append(best_match.get(key, None))
        
        df_pred_filename["score"] = max_scores
        for key, values in max_coords.items():
            df_pred_filename[key] = values
        return df_pred_filename
    except Exception as e:
        print(f"Error in comparison: {e}")
        return pd.DataFrame()

def concat_frames(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate predicted and ground truth datasets with IOU scores.

    Args:
        df_pred (pd.DataFrame): DataFrame with predicted bounding boxes.
        df_gt (pd.DataFrame): DataFrame with ground truth bounding boxes.

    Returns:
        pd.DataFrame: Concatenated DataFrame with calculated IOU scores.
    """
    try:
        df_pred = df_pred.rename(columns={"class": "class_pred", "xmin": "xmin_pred", "ymin": "ymin_pred", "xmax": "xmax_pred", "ymax": "ymax_pred"})
        df_gt = df_gt.rename(columns={"class": "class_gt", "xmin": "xmin_gt", "ymin": "ymin_gt", "xmax": "xmax_gt", "ymax": "ymax_gt"})
        
        frames = [comparison(df_pred[df_pred.filename == element], df_gt[df_gt.filename == element]) for element in df_pred.filename.unique()]
        return pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating frames: {e}")
        return pd.DataFrame()

def box_plot_iou(df_concat: pd.DataFrame, accuracy_txt_path: str = None) -> go.Figure:
    """
    Generate a box plot for IOU scores.

    Args:
        df_concat (pd.DataFrame): DataFrame with IOU scores.
        accuracy_txt_path (str, optional): Path to save accuracy percentages.

    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        fig = px.box(
            df_concat, y="score", points="all",
            title="IOU Scores Distribution", labels={"score": "IOU Score"}
        )
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(title='Classes', tickfont=dict(size=35)),
            yaxis=dict(title='IOU Score', tickfont=dict(size=35)),
            title=dict(text='Label Detection IOU Scores Distribution', font=dict(size=40), x=0.5)
        )
        
        if accuracy_txt_path:
            accuracy_percentage = df_concat['score'].mean() * 100
            Path(accuracy_txt_path).write_text(f"Overall Accuracy Percentage: {accuracy_percentage:.2f}%\n")
        
        return fig
    except Exception as e:
        print(f"Error generating box plot: {e}")
        return go.Figure()
