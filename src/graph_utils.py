"""
Graph construction utilities for the transaction–entity multipartite graph.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def build_bipartite_entity_graph(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    include_transaction_nodes: bool = True,
) -> nx.Graph:
    """
    Build a multipartite graph:
      transaction nodes  --  user / device / IP nodes

    For large data we optionally subsample.  The full 151k-row graph is
    feasible in NetworkX for degree/centrality calculations but may be
    slow for embeddings; callers should decide.
    """
    data = df if max_rows is None else df.head(max_rows)
    G = nx.Graph()

    for idx, r in data.iterrows():
        if include_transaction_nodes:
            tx = f"tx::{idx}"
            G.add_node(tx, node_type="transaction", purchase_value=r.get("purchase_value", 0))
        else:
            tx = None

        for col, ntype in [("user_id", "user"), ("device_id", "device"), ("ip_address", "ip")]:
            if col not in data.columns or pd.isna(r[col]):
                continue
            node = f"{ntype}::{r[col]}"
            G.add_node(node, node_type=ntype)
            if tx is not None:
                G.add_edge(tx, node, relation=ntype)
            # Also connect co-occurring entities through a soft link if desired
            # (kept simple here)

    return G


def entity_degree_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight degree / frequency features without materialising the full graph.
    Equivalent to 1-hop degree of each entity node.
    """
    out = pd.DataFrame(index=df.index)
    for col, name in [("user_id", "user"), ("device_id", "device"), ("ip_address", "ip")]:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            out[f"{name}_degree"] = df[col].map(counts).astype(float)
        else:
            out[f"{name}_degree"] = 0.0
    return out


def shared_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many other entities share the same device / IP etc.
    Useful proxy for collusion rings without expensive all-pairs computation.
    """
    x = df.copy()
    # Users sharing a device
    if "device_id" in x.columns and "user_id" in x.columns:
        device_user_counts = x.groupby("device_id")["user_id"].nunique()
        x["users_on_device"] = x["device_id"].map(device_user_counts).astype(float)
    else:
        x["users_on_device"] = 1.0

    if "ip_address" in x.columns and "user_id" in x.columns:
        ip_user_counts = x.groupby("ip_address")["user_id"].nunique()
        x["users_on_ip"] = x["ip_address"].map(ip_user_counts).astype(float)
    else:
        x["users_on_ip"] = 1.0

    if "device_id" in x.columns and "ip_address" in x.columns:
        device_ip_counts = x.groupby("device_id")["ip_address"].nunique()
        x["ips_on_device"] = x["device_id"].map(device_ip_counts).astype(float)
    else:
        x["ips_on_device"] = 1.0

    return x[["users_on_device", "users_on_ip", "ips_on_device"]]


def sample_subgraph_for_viz(
    df: pd.DataFrame,
    n_transactions: int = 200,
    seed: int = 42,
) -> nx.Graph:
    """Small random sample suitable for interactive visualisation."""
    sample = df.sample(n=min(n_transactions, len(df)), random_state=seed)
    return build_bipartite_entity_graph(sample, include_transaction_nodes=True)
