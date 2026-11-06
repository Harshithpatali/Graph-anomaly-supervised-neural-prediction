from pathlib import Path
import json
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.customer_intelligence import build_customer_profiles
from src.customer_segmentation import segment_customers
from src.customer_risk import build_behavioral_risk
from src.unsupervised_anomalies import score_anomalies
from src.graph_intelligence import build_entity_graph, graph_entity_metrics, community_summary


def main():
    source = ROOT / 'data/raw/Fraud_Data.csv'
    output = ROOT / 'reports/customer_intelligence'
    output.mkdir(parents=True, exist_ok=True)
    tx = pd.read_csv(source)
    required = {'user_id', 'purchase_time', 'purchase_value', 'device_id', 'ip_address'}
    missing = required.difference(tx.columns)
    if missing:
        raise ValueError(f'Missing source fields: {sorted(missing)}')
    tx = tx.drop(columns=['class'], errors='ignore')
    tx['purchase_time'] = pd.to_datetime(tx['purchase_time'], errors='coerce', utc=True)
    tx = tx.dropna(subset=['user_id', 'purchase_time', 'purchase_value'])
    profiles = build_customer_profiles(tx)
    risk = build_behavioral_risk(profiles)
    segmented, cluster_metrics, segment_meta = segment_customers(risk)
    customers, customer_meta = score_anomalies(segmented)
    graph = build_entity_graph(tx, max_transactions=50000)
    entities = graph_entity_metrics(graph)
    communities = community_summary(graph)
    tx_features = tx.copy()
    tx_features['purchase_hour'] = tx_features['purchase_time'].dt.hour
    tx_features['purchase_dayofweek'] = tx_features['purchase_time'].dt.dayofweek
    for col in ('user_id', 'device_id', 'ip_address'):
        tx_features[f'{col}_frequency'] = tx_features[col].map(tx_features[col].value_counts())
    cols = [c for c in ['purchase_value', 'age', 'purchase_hour', 'purchase_dayofweek', 'user_id_frequency', 'device_id_frequency', 'ip_address_frequency'] if c in tx_features]
    transactions, tx_meta = score_anomalies(tx_features, cols)
    profiles.to_parquet(output / 'customer_profiles.parquet', index=False)
    customers.to_parquet(output / 'customer_intelligence.parquet', index=False)
    transactions.head(10000).to_parquet(output / 'top_transaction_anomalies.parquet', index=False)
    cluster_metrics.to_csv(output / 'segmentation_metrics.csv', index=False)
    pd.DataFrame(segment_meta['segment_summary']).to_csv(output / 'segment_summary.csv', index=False)
    entities.to_csv(output / 'graph_entity_metrics.csv', index=False)
    communities.to_csv(output / 'graph_communities.csv', index=False)
    metadata = {'project': 'Graph-Based Customer Intelligence', 'label_used_for_features_or_training': False,
                'transactions': len(tx), 'customers': len(profiles), 'graph_nodes': graph.number_of_nodes(),
                'graph_edges': graph.number_of_edges(), 'segmentation': segment_meta['metadata'],
                'customer_anomaly': customer_meta, 'transaction_anomaly': tx_meta,
                'limitations': ['Risk index is a relative heuristic, not fraud probability.',
                                'Graph uses the first 50,000 input transactions.',
                                'Entity frequencies are full-snapshot descriptive features, not point-in-time predictions.']}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
