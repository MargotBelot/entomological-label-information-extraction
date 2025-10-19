# Import third-party libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Enhanced consistent style parameters for all evaluation visualizations
FONT_SIZE_TITLE = 22
FONT_SIZE_SUBTITLE = 18
FONT_SIZE_LABEL = 18
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_ANNOTATION = 18
FONT_SIZE_HEATMAP_ANNOT = 18
FIGURE_SIZE = (12, 10)
FIGURE_SIZE_CONFUSION = (12, 10)
FIGURE_SIZE_IOU = (10, 8)
DPI = 300
LINEWIDTH = 2.5
MARKERSIZE = 8

# Color palette for consistent styling across all evaluations
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
COLORMAP_CONFUSION = 'Blues'
COLORMAP_HEATMAP = 'viridis'

# Consistent padding and spacing
TITLE_PAD = 25
LABEL_PAD = 15
GRID_ALPHA = 0.3

def set_matplotlib_style():
    """Set consistent matplotlib style for all evaluation plots with enhanced readability."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': FONT_SIZE_TICK,
        'font.family': 'DejaVu Sans',  # Consistent font family
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'axes.titlepad': TITLE_PAD,
        'axes.labelpad': LABEL_PAD,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': GRID_ALPHA,
        'grid.linewidth': 0.5,
        'lines.linewidth': LINEWIDTH,
        'lines.markersize': MARKERSIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

def apply_consistent_style_to_ax(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Apply consistent styling to a matplotlib axes object.
    
    Args:
        ax: matplotlib axes object
        title (str): Title for the plot
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=TITLE_PAD, weight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    
    # Apply consistent tick formatting
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

def create_combined_violin_plot(data_dict: dict, score_name: str, file_name: str, 
                               title: str = None, y_limit: float = None) -> None:
    """
    Create and save a combined violin plot comparing multiple datasets.
    
    Args:
        data_dict (dict): Dictionary with dataset names as keys and score lists as values
        score_name (str): Name of the score (e.g., "CER" or "WER")
        file_name (str): Path to save the plot image
        title (str, optional): Custom title for the plot
        y_limit (float, optional): Maximum y-axis limit to prevent squishing
    """
    set_matplotlib_style()
    
    # Prepare data for plotting
    plot_data = []
    for dataset, scores in data_dict.items():
        for score in scores:
            plot_data.append({'Dataset': dataset, score_name: score})
    
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create violin plot with better styling
    ax = sns.violinplot(data=df, x='Dataset', y=score_name, inner="box", 
                       palette=COLORS[:len(data_dict)], cut=1)
    
    # Add mean and median lines for each dataset
    for i, (dataset, scores) in enumerate(data_dict.items()):
        mean_val = np.mean(scores)
        median_val = np.median(scores)
        
        # Add horizontal lines for mean and median within each violin
        x_pos = i
        width = 0.4
        plt.plot([x_pos - width/2, x_pos + width/2], [mean_val, mean_val], 
                'r--', linewidth=2, alpha=0.8)
        plt.plot([x_pos - width/2, x_pos + width/2], [median_val, median_val], 
                'g-', linewidth=2, alpha=0.8)
    
    # Set title and labels
    if title is None:
        title = f"Distribution of {score_name} Across Datasets"
    
    plt.title(title, fontsize=FONT_SIZE_TITLE, pad=20)
    plt.xlabel("Dataset", fontsize=FONT_SIZE_LABEL)
    plt.ylabel(score_name, fontsize=FONT_SIZE_LABEL)  # Fixed: use score_name instead of "Density"
    
    # Set y-axis limit if specified to prevent squishing
    if y_limit is not None:
        current_ylim = plt.ylim()
        plt.ylim(0, min(y_limit, current_ylim[1]))
    
    # Add legend for mean and median
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean'),
                      Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Median')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE_LEGEND)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Combined violin plot saved as {file_name}")

def create_single_violin_plot(data: list, score_name: str, file_name: str, 
                             dataset_name: str = "", y_limit: float = None) -> None:
    """
    Create and save a single violin plot with enhanced consistent styling.
    
    Args:
        data (list): List of numerical scores to visualize
        score_name (str): Name of the score (e.g., "CER" or "WER")
        file_name (str): Path to save the plot image
        dataset_name (str): Name of the dataset for title
        y_limit (float, optional): Maximum y-axis limit to prevent squishing
    """
    set_matplotlib_style()
    
    plt.figure(figsize=FIGURE_SIZE)
    df = pd.DataFrame(data, columns=[score_name])
    
    # Create violin plot with enhanced styling
    ax = sns.violinplot(data=df, y=score_name, inner="box", cut=1, 
                       palette=[COLORS[0]], width=0.7, linewidth=LINEWIDTH/2)
    
    # Calculate statistics
    mean_val = df[score_name].mean()
    median_val = df[score_name].median()
    std_val = df[score_name].std()
    
    # Add mean and median lines with consistent styling
    plt.axhline(mean_val, color=COLORS[3], linestyle="--", linewidth=LINEWIDTH, alpha=0.8,
                label=f"Mean: {mean_val:.3f}")
    plt.axhline(median_val, color=COLORS[2], linestyle="-", linewidth=LINEWIDTH, alpha=0.8,
                label=f"Median: {median_val:.3f}")
    
    # Enhanced title and labels
    title = f"Distribution of {score_name}"
    if dataset_name:
        title += f" - {dataset_name}"
    
    apply_consistent_style_to_ax(ax, title=title, ylabel=score_name)
    
    # Set y-axis limit if specified
    if y_limit is not None:
        current_ylim = plt.ylim()
        plt.ylim(0, min(y_limit, current_ylim[1]))
    
    # Enhanced legend with statistics
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS[3], linestyle='--', linewidth=LINEWIDTH, 
                  label=f'Mean: {mean_val:.3f}'),
        plt.Line2D([0], [0], color=COLORS[2], linestyle='-', linewidth=LINEWIDTH, 
                  label=f'Median: {median_val:.3f}'),
        plt.Line2D([0], [0], color='black', linestyle=':', linewidth=LINEWIDTH, 
                  label=f'Std Dev: {std_val:.3f}')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE_LEGEND,
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Enhanced violin plot saved as {file_name}")

def create_elbow_plot(x_values: list, y_values: list, file_name: str, 
                     x_label: str = "Number of Clusters", y_label: str = "SSE",
                     title: str = "Elbow Method for Optimal Clusters",
                     optimal_point: int = None) -> None:
    """
    Create and save an elbow plot with correct labels and improved styling.
    
    Args:
        x_values (list): X-axis values (should be number of clusters)
        y_values (list): Y-axis values (SSE values)
        file_name (str): Path to save the plot image
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Plot title
        optimal_point (int, optional): Mark the optimal number of clusters
    """
    set_matplotlib_style()
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create the elbow plot
    plt.plot(x_values, y_values, 'bo-', linewidth=2, markersize=8, color=COLORS[0])
    
    # Mark optimal point if provided
    if optimal_point is not None and optimal_point in x_values:
        idx = x_values.index(optimal_point)
        plt.plot(optimal_point, y_values[idx], 'ro', markersize=12, 
                label=f'Optimal: {optimal_point} clusters')
        plt.legend(fontsize=FONT_SIZE_LEGEND)
    
    plt.title(title, fontsize=FONT_SIZE_TITLE, pad=20)
    plt.xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    plt.ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Elbow plot saved as {file_name}")

def create_improved_iou_plot(df_concat: pd.DataFrame, file_name: str, 
                            accuracy_threshold: float = 0.8) -> go.Figure:
    """
    Create an enhanced IOU scores box plot with consistent styling and readable fonts.
    
    Args:
        df_concat (pd.DataFrame): DataFrame with IOU scores
        file_name (str): Path to save the plot
        accuracy_threshold (float): Threshold for accuracy calculation
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.box(
        df_concat,
        y="score",
        points="all",
        title="Label Detection IOU Scores Distribution",
        color_discrete_sequence=[COLORS[0]]
    )
    
    # Add threshold line with enhanced styling and visibility
    fig.add_hline(
        y=accuracy_threshold,
        line_dash="dash",
        line_color="red",  # More prominent red color for threshold
        line_width=LINEWIDTH + 1,  # Slightly thicker line
        annotation_text=f"<b>Accuracy Threshold: {accuracy_threshold}</b>",
        annotation_font=dict(size=FONT_SIZE_ANNOTATION + 2, color='red', family='Arial Black'),
        annotation_position="top right"
    )
    
    # Enhanced styling with bold fonts for publication quality
    fig.update_layout(
        width=1200,
        height=800,
        title=dict(
            text="<b>Label Detection IOU Scores Distribution</b>",
            font=dict(size=FONT_SIZE_TITLE + 2, family='Arial Black', color='black'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=FONT_SIZE_TICK + 2, family='Arial', color='black'),
            titlefont=dict(size=FONT_SIZE_LABEL + 2, family='Arial Black', color='black'),
            showgrid=True,
            gridcolor=f'rgba(128,128,128,{GRID_ALPHA})',
            linecolor='black',
            linewidth=2
        ),
        yaxis=dict(
            title="<b>IOU Score</b>",
            tickfont=dict(size=FONT_SIZE_TICK + 2, family='Arial', color='black'),
            titlefont=dict(size=FONT_SIZE_LABEL + 2, family='Arial Black', color='black'),
            range=[0, 1.1],
            showgrid=True,
            gridcolor=f'rgba(128,128,128,{GRID_ALPHA})',
            linecolor='black',
            linewidth=2
        ),
        font=dict(size=FONT_SIZE_TICK + 2, family='Arial', color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=60, t=120, b=100)  # Increased margins for better spacing
    )
    
    # Calculate statistics
    accuracy_percentage = (df_concat["score"] >= accuracy_threshold).mean() * 100
    mean_iou = df_concat["score"].mean()
    std_iou = df_concat["score"].std()
    median_iou = df_concat["score"].median()
    
    # Enhanced annotation with bold key statistics for publication
    stats_text = (
        f"<b><span style='font-size:{FONT_SIZE_ANNOTATION + 2}px'>Statistics:</span></b><br>"
        f"<b>Accuracy (≥{accuracy_threshold}): {accuracy_percentage:.1f}%</b><br>"
        f"<b>Mean IOU: {mean_iou:.3f}</b><br>"
        f"<b>Median IOU: {median_iou:.3f}</b>"
    )
    
    fig.add_annotation(
        x=0.02,  # Slightly away from edge
        y=0.18,  # Positioned higher for better visibility
        text=stats_text,
        showarrow=False,
        font=dict(size=FONT_SIZE_ANNOTATION + 2, family='Arial Black', color='black'),
        bgcolor="rgba(255,255,255,0.98)",  # Nearly opaque background
        bordercolor="black",
        borderwidth=3,  # Even thicker border for publication
        borderpad=12,   # More generous padding
        align="left",
        width=200,      # Fixed width for consistent appearance
        height=120      # Fixed height for better layout
    )
    
    # Update traces for publication-quality styling and visibility
    fig.update_traces(
        marker=dict(
            size=10,  # Larger points for better visibility in publication
            opacity=0.9, 
            line=dict(width=2, color='black')  # Thicker black outlines
        ),
        line=dict(width=LINEWIDTH + 1),  # Thicker box lines
        fillcolor='rgba(31,119,180,0.4)',  # Slightly more opaque fill
        boxpoints='outliers',  # Show outliers only for cleaner appearance
        jitter=0.3,
        pointpos=0,
        # Enhanced box styling for publication
        whiskerwidth=0.5,  # Wider whiskers (max valid value)
        notchwidth=0.4,    # Wider notches within valid range [0, 0.5]
    )
    
    # Save the plot with enhanced resolution
    fig.write_html(file_name.replace('.png', '.html'))
    fig.write_image(file_name, width=1200, height=800)
    
    print(f"Enhanced IOU plot saved as {file_name}")
    return fig

def create_confusion_matrix_plot(confusion_matrix: np.ndarray, class_labels: list,
                                file_name: str, title: str = "Confusion Matrix", 
                                ylabel: str = "True Labels", normalize: bool = False,
                                colormap: str = None) -> None:
    """
    Create an enhanced confusion matrix plot with consistent styling and readable text.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix data
        class_labels (list): Labels for classes
        file_name (str): Path to save the plot
        title (str): Plot title
        ylabel (str): Y-axis label (default: "True Labels")
        normalize (bool): Whether to show normalized values (default: False)
        colormap (str): Colormap to use (default: None uses COLORMAP_CONFUSION)
    """
    set_matplotlib_style()
    
    plt.figure(figsize=FIGURE_SIZE_CONFUSION)
    
    # Normalize confusion matrix if requested
    if normalize:
        cm_display = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cbar_label = 'Proportion'
    else:
        cm_display = confusion_matrix
        fmt = 'd'
        cbar_label = 'Count'
    
    # Use custom colormap if provided, otherwise use default
    cmap_to_use = colormap if colormap else COLORMAP_CONFUSION
    
    # Create heatmap with enhanced styling
    sns.heatmap(cm_display, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap_to_use,
                xticklabels=class_labels, 
                yticklabels=class_labels,
                annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'weight': 'bold'}, 
                cbar_kws={'label': cbar_label, 'shrink': 0.8},
                square=True,
                linewidths=0.5,
                linecolor='white')
    
    # Enhanced title and labels with consistent styling
    plt.title(title, fontsize=FONT_SIZE_TITLE, pad=TITLE_PAD, weight='bold')
    plt.xlabel('Predicted Labels', fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    plt.ylabel(ylabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    
    # Enhanced tick formatting
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE_TICK)
    plt.yticks(rotation=0, fontsize=FONT_SIZE_TICK)
    
    # Adjust colorbar label font size
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Enhanced confusion matrix saved as {file_name}")

def create_combined_wer_cer_violin_plot(wer_data: list, cer_data: list, file_name: str, 
                                       dataset_name: str = "", y_limit: float = 4.0) -> None:
    """
    Create an overlaid violin plot showing both WER and CER distributions in the same plot area.
    Addresses reviewer comments M5: merge figures 7&8, proper y-axis labels, controlled range.
    
    Args:
        wer_data (list): List of WER scores
        cer_data (list): List of CER scores
        file_name (str): Path to save the plot image
        dataset_name (str): Name of the dataset for title
        y_limit (float): Maximum y-axis limit (default: 4.0 as suggested by reviewer)
    """
    set_matplotlib_style()
    
    # Create figure with publication-quality styling
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]))
    
    # Create violin plots for both WER and CER overlaid in the same space
    import numpy as np
    try:
        from scipy.stats import gaussian_kde
        scipy_available = True
    except ImportError:
        print("Warning: scipy not available, using simplified violin plot")
        scipy_available = False
    
    # Calculate statistics for both metrics
    wer_mean = np.mean(wer_data)
    wer_median = np.median(wer_data)
    cer_mean = np.mean(cer_data)
    cer_median = np.median(cer_data)
    
    if scipy_available:
        # Create kernel density estimates for both datasets
        if len(wer_data) > 1:
            wer_kde = gaussian_kde(wer_data)
            wer_y = np.linspace(0, min(y_limit, max(wer_data) * 1.1), 100)
            wer_density = wer_kde(wer_y)
            # Normalize density for violin plot width
            wer_density = wer_density / np.max(wer_density) * 0.4
            
            # Create WER violin (left side)
            ax.fill_betweenx(wer_y, -wer_density, 0, alpha=0.7, color=COLORS[0], 
                            label='WER Distribution', edgecolor='black', linewidth=1)
        
        if len(cer_data) > 1:
            cer_kde = gaussian_kde(cer_data)
            cer_y = np.linspace(0, min(y_limit, max(cer_data) * 1.1), 100)
            cer_density = cer_kde(cer_y)
            # Normalize density for violin plot width
            cer_density = cer_density / np.max(cer_density) * 0.4
            
            # Create CER violin (right side)
            ax.fill_betweenx(cer_y, 0, cer_density, alpha=0.7, color=COLORS[1], 
                            label='CER Distribution', edgecolor='black', linewidth=1)
    else:
        # Fallback: create simple box plots side by side
        wer_positions = [-0.2]
        cer_positions = [0.2]
        
        bp1 = ax.boxplot(wer_data, positions=wer_positions, widths=0.3, 
                        patch_artist=True, labels=[''])
        bp2 = ax.boxplot(cer_data, positions=cer_positions, widths=0.3, 
                        patch_artist=True, labels=[''])
        
        # Color the boxes
        bp1['boxes'][0].set_facecolor(COLORS[0])
        bp1['boxes'][0].set_alpha(0.7)
        bp2['boxes'][0].set_facecolor(COLORS[1])
        bp2['boxes'][0].set_alpha(0.7)
    
    # Add individual data points for transparency
    # WER points (left side)
    wer_jitter = np.random.normal(-0.1, 0.03, len(wer_data))
    ax.scatter(wer_jitter, wer_data, alpha=0.6, color=COLORS[0], s=30, edgecolors='black', linewidth=0.5)
    
    # CER points (right side)
    cer_jitter = np.random.normal(0.1, 0.03, len(cer_data))
    ax.scatter(cer_jitter, cer_data, alpha=0.6, color=COLORS[1], s=30, edgecolors='black', linewidth=0.5)
    
    # Add mean and median lines
    ax.axhline(wer_mean, xmin=0, xmax=0.5, color=COLORS[3], linestyle='--', 
              linewidth=LINEWIDTH, alpha=0.9, label=f'WER Mean: {wer_mean:.3f}')
    ax.axhline(cer_mean, xmin=0.5, xmax=1, color=COLORS[4], linestyle='--', 
              linewidth=LINEWIDTH, alpha=0.9, label=f'CER Mean: {cer_mean:.3f}')
    
    # Add median lines (solid)
    ax.axhline(wer_median, xmin=0, xmax=0.5, color=COLORS[3], linestyle='-', 
              linewidth=LINEWIDTH-1, alpha=0.7)
    ax.axhline(cer_median, xmin=0.5, xmax=1, color=COLORS[4], linestyle='-', 
              linewidth=LINEWIDTH-1, alpha=0.7)
    
    # Set title with proper formatting
    title = "WER vs CER Error Rate Comparison with Tesseract OCR"
    if dataset_name:
        title += f" - {dataset_name}"
    
    apply_consistent_style_to_ax(ax, title=title, xlabel="Distribution Comparison", ylabel="Error Rate")
    
    # Set y-axis limit as suggested by reviewer (0 to 4 for comparison across datasets)
    ax.set_ylim(0, y_limit)
    ax.set_xlim(-0.6, 0.6)  # Center the violins
    
    # Remove x-axis ticks since we have overlaid distributions
    ax.set_xticks([])
    
    # Add subtle grid for better readability
    ax.grid(True, alpha=GRID_ALPHA, axis='y')
    
    # Add statistics text box
    stats_text = (
        f"WER: Mean={wer_mean:.3f}, Median={wer_median:.3f} (n={len(wer_data)})\n"
        f"CER: Mean={cer_mean:.3f}, Median={cer_median:.3f} (n={len(cer_data)})"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=FONT_SIZE_LEGEND, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                     edgecolor='black', linewidth=1))
    
    # Enhanced legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[0], alpha=0.7, edgecolor='black', label='WER Distribution'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[1], alpha=0.7, edgecolor='black', label='CER Distribution'),
        plt.Line2D([0], [0], color=COLORS[3], linestyle='--', linewidth=LINEWIDTH, label='WER Mean'),
        plt.Line2D([0], [0], color=COLORS[4], linestyle='--', linewidth=LINEWIDTH, label='CER Mean')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE_LEGEND,
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Enhanced overlaid WER/CER violin plot saved as {file_name}")
    print(f"WER and CER distributions combined in the same plot area with transparency")
    print(f"Addressed reviewer M5 comments: merged figures, proper labels, controlled range")

def create_clustering_metrics_plot(thresholds: list, metrics_data: dict, 
                                  file_name: str, dataset_name: str = "") -> None:
    """
    Create clustering metrics plot with improved styling and corrected labels.
    
    Args:
        thresholds (list): Similarity thresholds
        metrics_data (dict): Dictionary with metric names as keys and values as lists
        file_name (str): Path to save the plot
        dataset_name (str): Name of dataset for title
    """
    set_matplotlib_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    metrics = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
    colors = [COLORS[0], COLORS[1], COLORS[2]]
    
    for i, (metric, ax, color) in enumerate(zip(metrics, axes, colors)):
        if metric in metrics_data:
            ax.plot(thresholds, metrics_data[metric], 'o-', linewidth=2, 
                   markersize=6, color=color)
            ax.set_title(f'{metric} - {dataset_name}', fontsize=FONT_SIZE_TITLE)
            ax.set_xlabel('Similarity Thresholds', fontsize=FONT_SIZE_LABEL)  # Corrected label
            ax.set_ylabel(metric, fontsize=FONT_SIZE_LABEL)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering metrics plot saved as {file_name}")