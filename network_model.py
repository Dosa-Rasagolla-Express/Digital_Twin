"""
network_model.py
================
NetworkX-based junction graph for the Smart Traffic Digital Twin.

Models 5 interconnected junctions and propagates traffic conditions
between neighbours (congestion spill-over effect).
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from typing import Dict, Optional


# ─────────────────────────────────────────────
# JUNCTION COORDINATES (for layout)
# ─────────────────────────────────────────────

JUNCTION_POS = {
    "Main Junction":  (0,  0),
    "North Junction": (0,  2),
    "South Junction": (0, -2),
    "East Junction":  (2,  0),
    "West Junction":  (-2, 0),
}

JUNCTION_EDGES = [
    ("Main Junction", "North Junction"),
    ("Main Junction", "South Junction"),
    ("Main Junction", "East Junction"),
    ("Main Junction", "West Junction"),
    ("North Junction", "East Junction"),
    ("South Junction", "West Junction"),
]

# Propagation factor: fraction of excess congestion passed to neighbours
PROPAGATION_FACTOR = 0.25
HIGH_THRESHOLD     = 20
MEDIUM_THRESHOLD   = 10


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_traffic_graph(junction_vehicles: Dict[str, int]) -> nx.Graph:
    """
    Build a NetworkX graph with vehicle-count node attributes.

    Parameters
    ----------
    junction_vehicles : {junction_name: vehicle_count}

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    # Add nodes
    for name, pos in JUNCTION_POS.items():
        vc = junction_vehicles.get(name, 0)
        cong = (
            "HIGH" if vc >= HIGH_THRESHOLD
            else "MEDIUM" if vc >= MEDIUM_THRESHOLD
            else "LOW"
        )
        G.add_node(
            name,
            vehicles=vc,
            congestion=cong,
            pos=pos,
        )

    # Add edges with weight = average vehicle count of endpoints
    for u, v in JUNCTION_EDGES:
        avg_vc = (
            junction_vehicles.get(u, 0) + junction_vehicles.get(v, 0)
        ) / 2
        G.add_edge(u, v, weight=avg_vc)

    return G


# ─────────────────────────────────────────────
# CONGESTION PROPAGATION
# ─────────────────────────────────────────────

def propagate_congestion(
    junction_vehicles: Dict[str, int]
) -> Dict[str, int]:
    """
    Apply a single round of congestion propagation.

    High-traffic junctions push a fraction of their excess
    vehicles onto connected neighbours.

    Returns updated vehicle-count dict.
    """
    G = build_traffic_graph(junction_vehicles)
    updated = dict(junction_vehicles)

    for node in G.nodes:
        vc = junction_vehicles.get(node, 0)
        if vc >= HIGH_THRESHOLD:
            excess    = vc - HIGH_THRESHOLD
            neighbors = list(G.neighbors(node))
            if neighbors:
                spill = int(excess * PROPAGATION_FACTOR / len(neighbors))
                for nb in neighbors:
                    updated[nb] = updated.get(nb, 0) + spill

    return updated


# ─────────────────────────────────────────────
# GRAPH VISUALIZER
# ─────────────────────────────────────────────

def _node_color(congestion: str) -> str:
    return {
        "LOW":    "#00DC5A",
        "MEDIUM": "#FFC800",
        "HIGH":   "#FF3C3C",
    }.get(congestion, "#888888")


def render_network_graph(
    junction_vehicles: Dict[str, int],
    title: str = "Junction Network — Traffic Propagation",
    figsize=(10, 7),
) -> BytesIO:
    """
    Render the NetworkX graph to a PNG image in memory.

    Returns
    -------
    BytesIO buffer containing the PNG image.
    """
    G = build_traffic_graph(junction_vehicles)
    pos = nx.get_node_attributes(G, "pos")

    node_colors  = [_node_color(G.nodes[n]["congestion"]) for n in G.nodes]
    node_sizes   = [500 + G.nodes[n]["vehicles"] * 40 for n in G.nodes]
    edge_weights = [G.edges[e]["weight"] for e in G.edges]
    edge_widths  = [max(1.0, w / 5) for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F4F5F7")
    ax.set_facecolor("#F4F5F7")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888888",
        width=edge_widths,
        alpha=0.8,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        edgecolors="#333333",
        linewidths=1.5
    )

    # Draw labels
    labels = {
        n: f"{n}\n{G.nodes[n]['vehicles']} veh"
        for n in G.nodes
    }
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_color="#000080",
        font_size=8,
        font_weight="bold",
    )

    # Edge weight labels
    edge_labels = {
        (u, v): f"{int(G.edges[u,v]['weight'])} avg"
        for u, v in G.edges
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=ax,
        font_color="#555555",
        font_size=7,
    )

    # Legend
    patches = [
        mpatches.Patch(color="#00DC5A", label="LOW congestion"),
        mpatches.Patch(color="#FFC800", label="MEDIUM congestion"),
        mpatches.Patch(color="#FF3C3C", label="HIGH congestion"),
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        facecolor="#FFFFFF",
        edgecolor="#D1D5DB",
        labelcolor="#333333",
        fontsize=9,
    )

    ax.set_title(title, color="#000080", fontsize=13, pad=15, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# NETWORK STATISTICS
# ─────────────────────────────────────────────

def network_stats(junction_vehicles: Dict[str, int]) -> dict:
    """
    Return basic graph-theoretic metrics for the junction network.
    """
    G = build_traffic_graph(junction_vehicles)
    centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    total_vc = sum(junction_vehicles.values())
    most_congested = max(junction_vehicles, key=junction_vehicles.get)
    most_central   = max(centrality, key=centrality.get)
    bottleneck      = max(betweenness, key=betweenness.get)

    return {
        "total_vehicles":   total_vc,
        "most_congested":   most_congested,
        "most_central":     most_central,
        "bottleneck":       bottleneck,
        "avg_degree":       round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "density":          round(nx.density(G), 3),
        "centrality":       centrality,
        "betweenness":      betweenness,
    }
