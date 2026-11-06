"""Entity graph construction and structural intelligence summaries."""
import networkx as nx
import pandas as pd


def build_entity_graph(transactions: pd.DataFrame, max_transactions: int = 50000) -> nx.Graph:
    required = {"user_id", "device_id", "ip_address"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing graph columns: {sorted(missing)}")
    graph = nx.Graph()
    for user_id, device_id, ip_address in transactions[["user_id", "device_id", "ip_address"]].head(max_transactions).itertuples(index=False, name=None):
        if pd.isna(user_id):
            continue
        user = f"user::{user_id}"
        graph.add_node(user, node_type="user")
        for value, kind in ((device_id, "device"), (ip_address, "ip")):
            if pd.notna(value):
                entity = f"{kind}::{value}"
                graph.add_node(entity, node_type=kind)
                graph.add_edge(user, entity, relation=kind)
    return graph


def graph_entity_metrics(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["entity_id", "node_type", "degree", "component_size", "betweenness"])
    component_size = {}
    for component in nx.connected_components(graph):
        for node in component:
            component_size[node] = len(component)
    between = nx.betweenness_centrality(graph, k=500, seed=42) if graph.number_of_nodes() > 500 else nx.betweenness_centrality(graph)
    rows = [{"entity_id": node, "node_type": graph.nodes[node].get("node_type", "unknown"),
             "degree": int(graph.degree(node)), "component_size": component_size[node],
             "betweenness": float(between[node])} for node in graph.nodes]
    return pd.DataFrame(rows).sort_values(["degree", "component_size"], ascending=False)


def community_summary(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for index, nodes in enumerate(sorted(nx.community.louvain_communities(graph, seed=42), key=len, reverse=True)):
        subgraph = graph.subgraph(nodes)
        rows.append({"community_id": index, "nodes": len(nodes), "edges": subgraph.number_of_edges(), "density": nx.density(subgraph)})
    return pd.DataFrame(rows, columns=["community_id", "nodes", "edges", "density"])
