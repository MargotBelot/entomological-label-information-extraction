# Import third-party libraries
import argparse
import os
import json
import string
import time
import warnings
import pandas as pd
import numpy as np
import gensim
from nltk import word_tokenize
from sklearn.manifold import TSNE
import plotly.express as px
from typing import Union, List, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'cluster_eval.py [-h] -c <path to cluster CSV> -gt <path to ground truth JSON> -o <output directory> -s <cluster size>'

    parser = argparse.ArgumentParser(
        description="Scatter plot of clusters using t-SNE.",
        add_help = False,
        usage = usage)
    
    parser.add_argument('-c', '--cluster_csv', required=True, help='Path to cluster CSV file')
    parser.add_argument('-gt', '--ground_truth', required=True, help='Path to ground truth JSON file')
    parser.add_argument('-o', '--out_dir', default='outputs', help='Directory to save output files')
    parser.add_argument('-s', '--cluster_size', type=int, default=1, help='Minimum cluster size to be plotted')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()


def is_word(token: str) -> bool:
    """
    Checks whether a token is a valid word (not punctuation or too short).
    Args:
        token (str): The token to check.
    Returns:
        bool: True if the token is a valid word, False otherwise.
    """
    return token not in string.punctuation and not token.isspace() and len(token) >= 3


def tokenize_text(labels: Union[List[Dict[str, str]], Dict[str, tuple[str, str]]], ground_truth: bool) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Tokenizes and lowercases text fields from labels.
    Args:
        labels (List[Dict[str, str]] or Dict[str, tuple[str, str]]): Labels to tokenize.
        ground_truth (bool): Whether the labels are ground truth data.
    Returns:
        List[Dict[str, Union[str, List[str]]]]: Tokenized labels with IDs.
    """
    tokenized = []
    for label in labels:
        text = label["text"] if ground_truth else label[1]
        tokens = [token.lower() for token in word_tokenize(text) if is_word(token)]
        if tokens:
            tokenized.append({"ID": label["ID"] if ground_truth else label[0], "tokens": tokens})
    return tokenized


def build_word_vectors(labels, ground_truth) -> tuple[gensim.models.Word2Vec, List[Dict[str, Union[str, List[str]]]]]:
    """
    Builds a Word2Vec model from the tokenized labels.
    Args:
        labels (List[Dict[str, str]] or Dict[str, tuple[str, str]]): Labels to build vectors from.
        ground_truth (bool): Whether the labels are ground truth data.
    Returns:
        tuple: A tuple containing the trained Word2Vec model and the tokenized labels.
    """
    tokenized = tokenize_text(labels, ground_truth)
    model = gensim.models.Word2Vec(
        [label["tokens"] for label in tokenized],
        vector_size=100, window=2, min_count=1, sg=1
    )
    return model, tokenized


def build_mean_label_vector(model, labels) -> tuple[Dict[str, np.ndarray], List[str]]:
    """
    Computes the mean vector for each label using the Word2Vec model.
    Also tracks labels that have no valid tokens (and thus no vector).
    Args:
        model (gensim.models.Word2Vec): The trained Word2Vec model.
        labels (List[Dict[str, List[str]]]): Tokenized labels with IDs.
    Returns:
        tuple: A tuple containing a dictionary of mean vectors and a list of skipped IDs.
    """
    vectors = {}
    skipped_ids = []

    for label in labels:
        tokens = [t for t in label["tokens"] if t in model.wv]
        if tokens:
            vectors[label["ID"]] = np.mean([model.wv[t] for t in tokens], axis=0)
        else:
            skipped_ids.append(label["ID"])
    return vectors, skipped_ids


def load_json(path: str) -> List[Dict[str, str]]:
    """
    Loads the ground truth JSON file.
    Args:
        path (str): Path to the JSON file.
    Returns:
        List[Dict[str, str]]: List of entries with "ID" and "text" fields.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry for entry in data if isinstance(entry, dict) and "ID" in entry and "text" in entry]


def load_cluster_csv(path: str) -> Dict[str, List[str]]:
    """
    Loads cluster assignments from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        Dict[str, List[str]]: Dictionary mapping label IDs to their cluster ID and transcript.
        Skips entries with missing "Transcript" or "Cluster_ID".
    """
    df = pd.read_csv(path, sep=';')
    return {
        str(row["ID"]).strip(): [str(row["Cluster_ID"]), str(row["Transcript"])]
        for _, row in df.iterrows()
        if not pd.isna(row["Transcript"]) and not pd.isna(row["Cluster_ID"])
    }


def plot_tsne(label_vectors: Dict[str, np.ndarray], clusters: Dict[str, List[str]], out_path: str, verbose: bool, skipped_ids: List[str]):
    """
    Generates and saves a t-SNE scatter plot with cluster coloring and hover text.
    Also includes skipped labels (no vectors) as a separate "No vector" cluster.
    Args:
        label_vectors (Dict[str, np.ndarray]): Dictionary of label IDs to their mean vectors.
        clusters (Dict[str, List[str]]): Dictionary mapping label IDs to their cluster ID and transcript.
        out_path (str): Path to save the t-SNE plot HTML file.
        verbose (bool): Whether to print verbose output.
        skipped_ids (List[str]): List of label IDs that had no valid tokens and thus no vector.
    Returns:
        plotly.graph_objects.Figure: The generated t-SNE plot.
    """
    # Add zero-vectors for skipped labels
    for sid in skipped_ids:
        label_vectors[sid] = np.zeros(100)  # Same vector size as Word2Vec
        clusters[sid] = ["No vector", "(no valid text)"]

    vectors = np.array(list(label_vectors.values()))
    ids = list(label_vectors.keys())
    cluster_ids = [clusters[i][0] if i in clusters else "Unassigned" for i in ids]
    transcripts = [clusters[i][1] if i in clusters else "(no text found)" for i in ids]

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(vectors)

    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df["label"] = ids
    df["cluster"] = cluster_ids
    df["text"] = transcripts

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["label", "text"],
        title="t-SNE Cluster Visualization for MfN_LEP_SEASIA"
    )
    fig.write_html(out_path)

    if verbose:
        print(f"t-SNE plot saved at {out_path}")
        print(f"Skipped {len(skipped_ids)} labels with no vector:")
        for i in skipped_ids[:10]:  # Show up to 10 skipped
            print(f"  - {i}")
        if len(skipped_ids) > 10:
            print(f"  ... and {len(skipped_ids) - 10} more.")

    return fig


def main(args):
    """
    Main entry point for clustering visualization.
    Loads data, trains embeddings, computes vectors, runs t-SNE, and saves plot.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.           
    Returns:
        None
    """
    os.makedirs(args.out_dir, exist_ok=True)
    start_time = time.time()

    try:
        if args.verbose:
            print(f"Loading data...")

        gt_data = load_json(args.ground_truth)
        cluster_data = load_cluster_csv(args.cluster_csv)

        # Normalize keys
        gt_dict = {entry["ID"].strip(): (None, entry["text"]) for entry in gt_data}
        cluster_data = {k.strip(): v for k, v in cluster_data.items()}

        # Merge cluster IDs into GT labels
        for label_id in gt_dict:
            if label_id in cluster_data:
                gt_dict[label_id] = (cluster_data[label_id][0], gt_dict[label_id][1])

        label_list = [{"ID": k, "text": v[1]} for k, v in gt_dict.items()]

        if args.verbose:
            print(f"Building Word2Vec model on {len(label_list)} labels...")

        model, tokenized = build_word_vectors(label_list, ground_truth=True)
        mean_vectors, skipped_ids = build_mean_label_vector(model, tokenized)

        out_path = os.path.join(args.out_dir, "cluster_visualization.html")
        plot_tsne(mean_vectors, cluster_data, out_path, args.verbose, skipped_ids)

        if args.verbose:
            print(f"Finished in {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
