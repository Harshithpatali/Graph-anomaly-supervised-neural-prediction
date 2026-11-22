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

def graph_quality_metrics(graph: nx.Graph) -> dict[str, object]:
    """Return deterministic structural diagnostics for the user-device-IP graph.

    These are descriptive graph-quality and connectivity measures, not fraud
    labels, predictive metrics, or evidence of operational detection value.
    """
    node_types = {
        node: graph.nodes[node].get("node_type", "unknown")
        for node in graph.nodes
    }
    type_counts = pd.Series(list(node_types.values())).value_counts().to_dict()
    edge_type_counts = {"user-device": 0, "user-ip": 0, "other": 0}
    invalid_edges = 0
    shared_device_nodes = 0
    shared_ip_nodes = 0
    users_linked_to_shared_device = set()
    users_linked_to_shared_ip = set()

    for left, right, attrs in graph.edges(data=True):
        pair = {node_types.get(left, "unknown"), node_types.get(right, "unknown")}
        relation = attrs.get("relation")
        if pair == {"user", "device"} and relation == "device":
            edge_type_counts["user-device"] += 1
        elif pair == {"user", "ip"} and relation == "ip":
            edge_type_counts["user-ip"] += 1
        else:
            edge_type_counts["other"] += 1
            invalid_edges += 1

    for node, kind in node_types.items():
        if kind == "device" and graph.degree(node) > 1:
            shared_device_nodes += 1
            users_linked_to_shared_device.update(
                neighbor for neighbor in graph.neighbors(node)
                if node_types.get(neighbor) == "user"
            )
        elif kind == "ip" and graph.degree(node) > 1:
            shared_ip_nodes += 1
            users_linked_to_shared_ip.update(
                neighbor for neighbor in graph.neighbors(node)
                if node_types.get(neighbor) == "user"
            )

    components = sorted(
        (len(component) for component in nx.connected_components(graph)),
        reverse=True,
    )
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    users = sum(kind == "user" for kind in node_types.values())
    return {
        "nodes": int(node_count),
        "edges": int(edge_count),
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "node_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "edge_type_counts": edge_type_counts,
        "invalid_edge_count": int(invalid_edges),
        "self_loop_count": int(nx.number_of_selfloops(graph)),
        "connected_components": int(len(components)),
        "largest_component_nodes": int(components[0]) if components else 0,
        "largest_component_share": float(components[0] / node_count) if node_count else 0.0,
        "isolated_nodes": int(len(list(nx.isolates(graph)))),
        "shared_device_entity_count": int(shared_device_nodes),
        "shared_ip_entity_count": int(shared_ip_nodes),
        "users_linked_to_shared_device_count": int(len(users_linked_to_shared_device)),
        "users_linked_to_shared_ip_count": int(len(users_linked_to_shared_ip)),
        "users_with_any_shared_entity_count": int(
            len(users_linked_to_shared_device | users_linked_to_shared_ip)
        ),
        "users_with_any_shared_entity_share": (
            float(len(users_linked_to_shared_device | users_linked_to_shared_ip) / users)
            if users else 0.0
        ),
        "interpretation": (
            "Descriptive connectivity diagnostics only; shared entities and high "
            "centrality require analyst review and are not fraud evidence."
        ),
    }

