# Updated plotting.py with sigmoid shading, triangular masking, and metadata-aware plotting

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from textwrap import wrap
from datetime import datetime
from html import escape

# Predefined colors for themes
THEME_COLORS = [
    "#F94144", "#F3722C", "#F9C74F", "#90BE6D", "#43AA8B",
    "#577590", "#277DA1", "#A05195", "#FF6F61", "#6A4C93"
]

def apply_sigmoid_shading(matrix, k):
    """Apply a sigmoid transform to matrix values, centered at 0.5."""
    return 1 / (1 + np.exp(-k * (matrix - 0.5)))

def plot_similarity_matrix(similarity_matrix, title="Lexical Similarity", zoom=100,
                           render_in_streamlit=True, overlay_lines=None,
                           theme_labels=None, theme_keywords=None,
                           k=None, height=None):
    # Apply triangular mask (lower triangle including diagonal)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    matrix_to_plot = np.ma.masked_array(similarity_matrix, mask=mask)

    # Apply sigmoid transform if k is set
    if k and k > 0:
        matrix_to_plot = apply_sigmoid_shading(matrix_to_plot, k=k)

    # Ensure values are within [0, 1]
    matrix_to_plot = np.clip(matrix_to_plot, 0, 1)

    num_segments = similarity_matrix.shape[0]
    base_size = max(6, min(12, num_segments * 0.4))
    zoom_factor = zoom / 100.0
    fig_size = base_size * zoom_factor

    fig, ax = plt.subplots(figsize=(fig_size, fig_size if height is None else height / 100))

    # Plot matrix (filled with 0s for masked upper triangle)
    if isinstance(matrix_to_plot, np.ma.MaskedArray):
        matrix_plot_data = np.tril(matrix_to_plot.filled(0))
    else:
        matrix_plot_data = np.tril(matrix_to_plot)

    cax = ax.imshow(matrix_plot_data, cmap='Greys', interpolation='none', origin='upper')
    ax.tick_params(labelsize=6, pad=2)
    tick_step = max(1, int(np.ceil(num_segments / 30)))
    ticks = np.arange(0, num_segments, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Text Segments")
    ax.set_ylabel("Text Segments")
    ax.set_title(title, fontsize=10)

    if overlay_lines:
        for item in overlay_lines:
            if isinstance(item, tuple) and len(item) == 2:
                idx, label = item
                if 0 <= idx < num_segments:
                    ax.axhline(idx, color='red', linestyle='--', linewidth=0.5)
                    ax.axvline(idx, color='red', linestyle='--', linewidth=0.5)

                    # Wrap the label text
                    wrapped_label = "\n".join(wrap(label, width=6))  # Wrap after 6 characters

                    # Add wrapped labels
                    ax.text(idx + 0.5, -1, wrapped_label, rotation=90, color='red', fontsize=6, ha='center', va='bottom')
                    ax.text(num_segments + 1, idx + 0.5, wrapped_label, rotation=0, color='red', fontsize=6, ha='left', va='center')

    if theme_labels is not None:
        for i in range(num_segments):
            theme_idx = theme_labels[i]
            color = THEME_COLORS[theme_idx % len(THEME_COLORS)]
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color=color, alpha=1.0, linewidth=0))

    if theme_keywords:
        legend_entries = []
        for cluster_id, words in theme_keywords.items():
            color = THEME_COLORS[cluster_id % len(THEME_COLORS)]
            label = f"Theme {cluster_id + 1}: {', '.join(words)}"
            legend_entries.append(plt.Line2D([0], [0], marker='s', color='w', label=label,
                                             markerfacecolor=color, markersize=10))
        ax.legend(handles=legend_entries, loc='upper right', fontsize=6, frameon=False)

    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.subplots_adjust(left=0.15, bottom=0.15, top=0.9, right=0.95)

    if render_in_streamlit:
        st.pyplot(fig)
    else:
        st.pyplot(fig)

def create_similarity_network_figure(
    similarity_matrix,
    node_labels=None,
    cluster_labels=None,
    theme_keywords=None,
    hover_snippets=None,
    threshold=0.6,
    max_edges=300,
    show_labels=False,
    title="Document Similarity Network"
):
    try:
        import networkx as nx
    except ImportError:
        return None, "NetworkX is required for the network view. Install it with `pip install networkx`."

    use_plotly = True
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        use_plotly = False
        go = None  # type: ignore

    if similarity_matrix is None or similarity_matrix.size == 0:
        return None, "No similarity data available to build a network."

    num_nodes = similarity_matrix.shape[0]
    node_labels = node_labels or [f"Segment {i}" for i in range(num_nodes)]

    # Build the edge list, keeping only the strongest connections.
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = similarity_matrix[i, j]
            if np.isnan(weight) or weight < threshold:
                continue
            edges.append((i, j, float(weight)))

    if not edges:
        return None, "No connections above the selected similarity threshold."

    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]

    graph = nx.Graph()
    for idx in range(num_nodes):
        label = node_labels[idx] if idx < len(node_labels) else f"Segment {idx}"
        cluster = cluster_labels[idx] if cluster_labels is not None and idx < len(cluster_labels) else None
        graph.add_node(idx, label=label, cluster=cluster)

    for i, j, weight in edges:
        graph.add_edge(i, j, weight=weight)

    if graph.number_of_edges() == 0:
        return None, "No connections above the selected similarity threshold."

    positions = nx.spring_layout(graph, weight="weight", seed=42, k=None)

    # Collect node attributes used for both backends.
    node_info = []
    for node_idx, data in graph.nodes(data=True):
        cluster = data.get("cluster")
        color = THEME_COLORS[cluster % len(THEME_COLORS)] if cluster is not None else "#888888"
        degree = graph.degree(node_idx)
        size = 14 + degree * 3
        label = data.get("label", f"Segment {node_idx}")
        cluster_text = ""
        if cluster is not None:
            cluster_text = f"Cluster {cluster + 1}"
            if theme_keywords:
                keywords = ", ".join(theme_keywords.get(cluster, []))
                if keywords:
                    cluster_text += f": {keywords}"
        else:
            cluster_text = "Cluster: N/A"
        preview = None
        if hover_snippets and node_idx < len(hover_snippets):
            preview = hover_snippets[node_idx]
        node_info.append({
            "index": node_idx,
            "label": label,
            "color": color,
            "degree": degree,
            "size": size,
            "cluster": cluster,
            "cluster_text": cluster_text,
            "preview": preview
        })

    edge_weights = [data.get("weight", 0.0) for _, _, data in graph.edges(data=True)]
    min_w = min(edge_weights) if edge_weights else 0.0
    max_w = max(edge_weights) if edge_weights else 0.0
    denom = max(max_w - min_w, 1e-6)
    edge_widths = [1.0 + 3.0 * ((w - min_w) / denom) for w in edge_weights]

    if use_plotly and go is not None:
        edge_traces = []
        for (edge_idx, (i, j, data)) in enumerate(graph.edges(data=True)):
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            weight = data.get("weight", 0.0)
            alpha = 0.25 + 0.55 * ((weight - min_w) / denom)
            edge_traces.append(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=edge_widths[edge_idx], color=f"rgba(90, 90, 90, {alpha:.2f})"),
                hoverinfo="none",
                showlegend=False
            ))

        hover_texts = []
        for n in node_info:
            lines = [f"<b>{escape(n['label'])}</b>", escape(n['cluster_text']), f"Connections: {n['degree']}"]
            preview = n.get("preview")
            if preview:
                preview_clean = " ".join(preview.split())
                wrapped_lines = wrap(preview_clean, width=80)
                preview_html = "<br>".join(escape(line) for line in wrapped_lines)
                preview_block = "<br><b>Preview</b><br>" + preview_html
                lines.append(preview_block)
            hover_texts.append("<br>".join(lines))

        node_trace = go.Scatter(
            x=[positions[n["index"]][0] for n in node_info],
            y=[positions[n["index"]][1] for n in node_info],
            mode="markers+text" if show_labels else "markers",
            text=[n["label"] for n in node_info] if show_labels else None,
            textposition="top center",
            hoverinfo="text",
            hovertext=hover_texts,
            customdata=[n["index"] for n in node_info],
            marker=dict(
                color=[n["color"] for n in node_info],
                size=[n["size"] for n in node_info],
                line=dict(width=0.8, color="#222222")
            ),
            showlegend=False
        )

        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        )

        return ("plotly", fig), None

    # Fallback: static matplotlib rendering if Plotly is unavailable.
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=edge_widths,
        edge_color="#BBBBBB",
        alpha=0.6
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=[n["color"] for n in node_info],
        node_size=[(n["size"] ** 2) * 2 for n in node_info],
        linewidths=0.8,
        edgecolors="#222222",
        alpha=0.95
    )
    if show_labels:
        label_map = {n["index"]: n["label"] for n in node_info}
        nx.draw_networkx_labels(graph, positions, labels=label_map, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")

    unique_clusters = sorted({n["cluster"] for n in node_info if n["cluster"] is not None})
    if unique_clusters:
        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"Cluster {c + 1}",
                markerfacecolor=THEME_COLORS[c % len(THEME_COLORS)],
                markersize=8
            )
            for c in unique_clusters
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8)

    fig.tight_layout()
    return ("matplotlib", fig), None

def generate_plot_pdf(similarity_matrix, title, preprocessing_opts, segmentation_settings,
                      overlay_lines=None, theme_labels=None, theme_keywords=None,
                      k=None):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Metadata page (standard A4 size)
        fig_metadata, ax_meta = plt.subplots(figsize=(8.5, 11))  # Standard letter size for metadata
        ax_meta.axis('off')

        # Add metadata (unchanged)
        cleaning = []
        if preprocessing_opts.get("lowercase"): cleaning.append("lowercased")
        if preprocessing_opts.get("remove_punctuation"): cleaning.append("punctuation removed")
        if preprocessing_opts.get("remove_stopwords"): cleaning.append("stopwords removed")
        if preprocessing_opts.get("strip_nonascii"): cleaning.append("non-ASCII characters removed")
        if preprocessing_opts.get("strip_scene_elements"): cleaning.append("scene/speaker elements removed")

        metadata_lines = [
            f"Input file: {title}",
            f"Semantic model used: {segmentation_settings.get('model_used', 'N/A')}",
            f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        preprocessing_str = f"Preprocessing: {', '.join(cleaning) or 'None'}"
        metadata_lines.extend(wrap(preprocessing_str, width=90))

        metadata_lines.append(f"Segmentation mode: {segmentation_settings['seg_type']}")
        if segmentation_settings["seg_type"] == "Evenly-sized content-word groups":
            metadata_lines.append(f"Target content words per segment: {segmentation_settings['content_word_target']}")
        else:
            metadata_lines.append(f"Units per segment: {segmentation_settings['segment_size']}")

        if segmentation_settings.get("use_padding"):
            metadata_lines.append("Padding: enabled (1-sentence look-ahead and look-behind)")
            metadata_lines.append(f"Stride: {segmentation_settings.get('stride', 1)}")
        else:
            metadata_lines.append("Padding: disabled")

        if k is not None:
            metadata_lines.append(f"Sigmoid visual transform applied (k = {k})")

        cutoff = preprocessing_opts.get("high_df_cutoff", 0)
        filtered_terms = preprocessing_opts.get("filtered_terms") or []
        if cutoff:
            metadata_lines.append(f"Corpus-specific filter: removed tokens appearing in ≥ {cutoff}% of segments.")
            if filtered_terms:
                preview = ", ".join(filtered_terms[:20])
                if len(filtered_terms) > 20:
                    preview += " …"
                metadata_lines.append(f"Filtered terms sample: {preview}")

        metadata_text = "VicPol Policy Tool Metadata\n\n" + "\n".join(metadata_lines)
        ax_meta.text(0.01, 0.99, metadata_text, verticalalignment='top', fontsize=12, wrap=True)
        pdf.savefig(fig_metadata)
        plt.close(fig_metadata)

        # --- Plot Page (A3 size) ---
        num_segments = similarity_matrix.shape[0]

        # Set A3 page size (landscape orientation)
        fig_width = 16.5  # A3 width in inches
        fig_height = 11.7  # A3 height in inches

        # Create the figure with A3 dimensions
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot the similarity matrix
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        masked_matrix = np.ma.masked_array(similarity_matrix, mask=mask)
        ax.imshow(masked_matrix, cmap='Greys', interpolation='none', origin='upper')

        # Set axis labels
        ax.set_xlabel("Text Segments")
        ax.set_ylabel("Text Segments")

        # Set tick marks
        tick_step = max(1, int(np.ceil(num_segments / 30)))
        ticks = np.arange(0, num_segments, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, rotation=90)
        ax.set_yticklabels(ticks)

        # --- Overlay marker lines ---
        if overlay_lines:
            for idx, label in overlay_lines:
                if 0 <= idx < num_segments:
                    ax.axhline(idx, color='red', linestyle='--', linewidth=0.5)
                    ax.axvline(idx, color='red', linestyle='--', linewidth=0.5)

                    # Wrap the label text
                    wrapped_label = "\n".join(wrap(label, width=15))  # Wrap after 15 characters

                    # Add wrapped labels
                    ax.text(idx + 0.5, -1, wrapped_label, rotation=90, color='red', fontsize=6, ha='center', va='bottom')
                    ax.text(num_segments + 1, idx + 0.5, wrapped_label, rotation=0, color='red', fontsize=6, ha='left', va='center')

        # Adjust margins to fit the content
        plt.subplots_adjust(left=0.10, bottom=0.15, top=0.9, right=0.95)

        # Add the title below the plot
        plt.figtext(0.5, -0.05, title, ha='center', fontsize=10, wrap=True)

        # Save the A3-sized figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()
