# similarity.py (BERT-only utilities)

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def cluster_segments(vectors, n_clusters=4):
    """Apply KMeans clustering to embedding vectors."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    return cluster_labels, kmeans


def label_clusters_by_tfidf(segments, cluster_labels, n_top=3):
    """Derive top-n TF-IDF keywords per cluster."""
    if not segments:
        return {}

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )
    X = tfidf.fit_transform(segments)
    terms = np.array(tfidf.get_feature_names_out())

    keywords = {}
    for cluster_id in sorted(set(cluster_labels)):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if not indices:
            keywords[cluster_id] = ["(no segments)"]
            continue
        cluster_matrix = X[indices]
        mean_scores = np.asarray(cluster_matrix.mean(axis=0)).flatten()
        top_idx = np.argsort(mean_scores)[::-1]

        ordered_terms = [terms[i] for i in top_idx if terms[i].isalpha()]
        if not ordered_terms:
            ordered_terms = [terms[i] for i in top_idx]

        keywords[cluster_id] = ordered_terms[:n_top] if ordered_terms else ["(no terms found)"]

    return keywords
