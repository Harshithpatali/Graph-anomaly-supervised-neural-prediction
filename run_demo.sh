#!/usr/bin/env bash
# Quick demo: build features, fit model, launch dashboard
set -e
echo ">>> Installing deps (if needed)..."
pip install -q -r requirements.txt
echo ">>> Building processed data & features (this may take a minute)..."
python -c "
import sys
sys.path.insert(0,'src')
from data_utils import load_raw, parse_timestamps, enrich_with_country, basic_clean, chronological_split, save_processed
from graph_features import build_feature_matrix
fraud, ipmap = load_raw()
fraud = basic_clean(fraud)
fraud = parse_timestamps(fraud)
fraud = enrich_with_country(fraud, ipmap)
train, val, test = chronological_split(fraud)
save_processed(fraud, 'fraud_enriched')
save_processed(train, 'train')
save_processed(val, 'val')
save_processed(test, 'test')
for name, part in [('X_train', train), ('X_val', val), ('X_test', test)]:
    X, _ = build_feature_matrix(part)
    X = X.assign(class=part['class'].values)
    save_processed(X, name)
print('Processed data ready.')
"
echo ">>> Launching Streamlit dashboard..."
streamlit run app/app.py --server.headless true
