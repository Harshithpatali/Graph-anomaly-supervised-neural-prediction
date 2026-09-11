"""Neural supervised fraud pipeline for Notebook 13.

Models:
    1. MLP baseline
    2. Embedding-based neural network

The module keeps feature engineering and inference in one place so that the
saved artifact can score raw transactions in the Streamlit application.
"""

from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "class"

BASE_REQUIRED = [
    "user_id", "signup_time", "purchase_time", "purchase_value",
    "device_id", "source", "browser", "sex", "age", "ip_address"
]

LOW_CARD = ["source", "browser", "sex", "ip_country"]
HIGH_CARD = ["user_id", "device_id", "ip_address"]

NUMERIC = [
    "purchase_value", "value_log1p", "age", "account_age_hours",
    "purchase_hour", "purchase_dow", "purchase_month", "purchase_day",
    "is_weekend", "user_frequency", "device_frequency", "ip_frequency",
    "user_device_frequency", "user_ip_frequency", "device_ip_frequency",
    "value_vs_train_median", "value_above_train_p95", "value_above_train_p99"
]


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Leakage-safe transaction feature engineering."""

    def __init__(self, ip_map=None):
        self.ip_map = ip_map

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.user_freq_ = X["user_id"].value_counts(dropna=False).to_dict()
        self.device_freq_ = X["device_id"].value_counts(dropna=False).to_dict()
        self.ip_freq_ = X["ip_address"].value_counts(dropna=False).to_dict()

        self.user_device_freq_ = (
            X.groupby(["user_id", "device_id"], dropna=False).size().to_dict()
        )
        self.user_ip_freq_ = (
            X.groupby(["user_id", "ip_address"], dropna=False).size().to_dict()
        )
        self.device_ip_freq_ = (
            X.groupby(["device_id", "ip_address"], dropna=False).size().to_dict()
        )

        values = pd.to_numeric(X["purchase_value"], errors="coerce")
        self.median_value_ = float(values.median())
        self.q95_value_ = float(values.quantile(0.95))
        self.q99_value_ = float(values.quantile(0.99))
        return self

    def _country(self, ips):
        if self.ip_map is None or len(self.ip_map) == 0:
            return pd.Series(["Unknown"] * len(ips), index=ips.index)

        m = (
            self.ip_map.sort_values("lower_bound_ip_address")
            .reset_index(drop=True)
        )
        starts = m["lower_bound_ip_address"].to_numpy()
        ends = m["upper_bound_ip_address"].to_numpy()
        countries = m["country"].astype(str).to_numpy()

        vals = pd.to_numeric(ips, errors="coerce").to_numpy()
        pos = np.searchsorted(starts, vals, side="right") - 1
        safe = np.clip(pos, 0, len(m) - 1)

        valid = (pos >= 0) & np.isfinite(vals) & (vals <= ends[safe])

        out = np.full(len(ips), "Unknown", dtype=object)
        out[valid] = countries[safe[valid]]
        return pd.Series(out, index=ips.index)

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        signup = pd.to_datetime(X["signup_time"], errors="coerce", utc=True)
        purchase = pd.to_datetime(X["purchase_time"], errors="coerce", utc=True)
        value = pd.to_numeric(X["purchase_value"], errors="coerce")

        out = pd.DataFrame(index=X.index)

        out["purchase_value"] = value
        out["value_log1p"] = np.log1p(value.clip(lower=0))
        out["age"] = pd.to_numeric(X["age"], errors="coerce")
        out["account_age_hours"] = (
            purchase - signup
        ).dt.total_seconds() / 3600

        out["purchase_hour"] = purchase.dt.hour
        out["purchase_dow"] = purchase.dt.dayofweek
        out["purchase_month"] = purchase.dt.month
        out["purchase_day"] = purchase.dt.day
        out["is_weekend"] = (purchase.dt.dayofweek >= 5).astype(float)

        out["user_frequency"] = X["user_id"].map(self.user_freq_).fillna(0)
        out["device_frequency"] = X["device_id"].map(self.device_freq_).fillna(0)
        out["ip_frequency"] = X["ip_address"].map(self.ip_freq_).fillna(0)

        out["user_device_frequency"] = [
            self.user_device_freq_.get(k, 0)
            for k in zip(X["user_id"], X["device_id"])
        ]
        out["user_ip_frequency"] = [
            self.user_ip_freq_.get(k, 0)
            for k in zip(X["user_id"], X["ip_address"])
        ]
        out["device_ip_frequency"] = [
            self.device_ip_freq_.get(k, 0)
            for k in zip(X["device_id"], X["ip_address"])
        ]

        out["value_vs_train_median"] = value / (self.median_value_ + 1e-9)
        out["value_above_train_p95"] = (value > self.q95_value_).astype(float)
        out["value_above_train_p99"] = (value > self.q99_value_).astype(float)

        out["ip_country"] = self._country(X["ip_address"])

        for col in ["source", "browser", "sex"]:
            out[col] = X[col].astype(str)

        for col in HIGH_CARD:
            out[col] = X[col].astype(str)

        return out


def fit_low_card_maps(features):
    maps = {}
    for col in LOW_CARD:
        values = features[col].astype(str).fillna("Unknown")
        maps[col] = {v: i for i, v in enumerate(sorted(values.unique()))}
    return maps


def encode_low_card(features, maps):
    encoded = []
    for col in LOW_CARD:
        s = features[col].astype(str).fillna("Unknown")
        encoded.append(s.map(maps[col]).fillna(-1).to_numpy())
    return np.column_stack(encoded).astype(np.int64)


def make_mlp_matrix(features, scaler=None, maps=None, fit=False):
    values = (
        features[NUMERIC]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if fit:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(values)
        maps = fit_low_card_maps(features)
    else:
        numeric = scaler.transform(values)

    blocks = [numeric.astype(np.float32)]

    encoded = encode_low_card(features, maps)

    for j, col in enumerate(LOW_CARD):
        n = len(maps[col])
        one_hot = np.zeros((len(features), n), dtype=np.float32)
        idx = encoded[:, j]
        valid = (idx >= 0) & (idx < n)
        rows = np.arange(len(features))[valid]
        one_hot[rows, idx[valid]] = 1.0
        blocks.append(one_hot)

    return np.hstack(blocks).astype(np.float32), scaler, maps


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64), dropout=0.25):
        super().__init__()
        layers = []
        d = input_dim

        for h in hidden:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            d = h

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _pos_weight(y):
    y = np.asarray(y)
    positives = max(int(y.sum()), 1)
    negatives = len(y) - positives
    return float(negatives / positives)


def train_mlp(X_train, y_train, X_val, y_val, config, seed=SEED, device=None):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPNet(
        X_train.shape[1],
        hidden=tuple(config["hidden"]),
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [_pos_weight(y_train)],
            dtype=torch.float32,
            device=device,
        )
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    x_val = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(config.get("epochs", 30)):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x_val)).cpu().numpy()

        ap = average_precision_score(y_val, probabilities)

        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= config.get("patience", 5):
                break

    model.load_state_dict(best_state)
    return model, best_ap, best_epoch, device


def predict_mlp(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X, dtype=torch.float32, device=device)
        )
        return torch.sigmoid(logits).cpu().numpy()


def fit_embedding_maps(features):
    maps = {}
    for col in HIGH_CARD + LOW_CARD:
        values = features[col].astype(str).fillna("Unknown")
        # 0 is reserved for unseen values at inference.
        maps[col] = {
            v: i + 1 for i, v in enumerate(sorted(values.unique()))
        }
    return maps


def encode_embeddings(features, maps):
    arrays = []
    for col in HIGH_CARD + LOW_CARD:
        arrays.append(
            features[col]
            .astype(str)
            .map(maps[col])
            .fillna(0)
            .to_numpy()
        )
    return np.column_stack(arrays).astype(np.int64)


def make_embedding_numeric(features, scaler=None, fit=False):
    values = (
        features[NUMERIC]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if fit:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(values)
    else:
        numeric = scaler.transform(values)

    return numeric.astype(np.float32), scaler


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        cardinalities,
        emb_dims,
        numeric_dim,
        hidden=(128, 64),
        dropout=0.25,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(cardinalities, emb_dims)
        ])

        total_dim = sum(emb_dims) + numeric_dim

        layers = []
        d = total_dim

        for h in hidden:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            d = h

        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, categorical, numeric):
        embedded = [
            embedding(categorical[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        combined = torch.cat(embedded + [numeric], dim=1)
        return self.mlp(combined).squeeze(1)


def train_embedding(
    Xc_train,
    Xn_train,
    y_train,
    Xc_val,
    Xn_val,
    y_val,
    cardinalities,
    config,
    seed=SEED,
    device=None,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingNet(
        cardinalities=cardinalities,
        emb_dims=config["emb_dims"],
        numeric_dim=Xn_train.shape[1],
        hidden=tuple(config["hidden"]),
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [_pos_weight(y_train)],
            dtype=torch.float32,
            device=device,
        )
    )

    dataset = TensorDataset(
        torch.tensor(Xc_train, dtype=torch.long),
        torch.tensor(Xn_train, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32)),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    xcv = torch.tensor(Xc_val, dtype=torch.long, device=device)
    xnv = torch.tensor(Xn_val, dtype=torch.float32, device=device)

    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(config.get("epochs", 35)):
        model.train()

        for categorical, numeric, yb in loader:
            categorical = categorical.to(device)
            numeric = numeric.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(categorical, numeric), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(xcv, xnv)
            ).cpu().numpy()

        ap = average_precision_score(y_val, probabilities)

        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= config.get("patience", 5):
                break

    model.load_state_dict(best_state)
    return model, best_ap, best_epoch, device


def predict_embedding(model, categorical, numeric, device):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                torch.tensor(categorical, dtype=torch.long, device=device),
                torch.tensor(numeric, dtype=torch.float32, device=device),
            )
        )
    return probabilities.cpu().numpy()


def evaluate_model(y, probabilities, threshold=0.5):
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y, probabilities, minimum_precision=None):
    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    if minimum_precision is not None:
        valid = np.where(
            precision[:-1] >= minimum_precision
        )[0]
        if len(valid):
            return float(
                thresholds[valid[np.argmax(recall[valid])]]
            )

    f1 = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )
    index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[index])


class NeuralFraudArtifact:
    """Saved raw-transaction inference wrapper."""

    def __init__(self, feature_transformer, model_type, model, metadata):
        self.feature_transformer = feature_transformer
        self.model_type = model_type
        self.model = model
        self.metadata = metadata

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        features = self.feature_transformer.transform(X)

        if self.model_type == "mlp":
            matrix, _, _ = make_mlp_matrix(
                features,
                scaler=self.metadata["scaler"],
                maps=self.metadata["low_maps"],
                fit=False,
            )
            probabilities = predict_mlp(
                self.model,
                matrix,
                self.metadata["device"],
            )
        else:
            numeric, _ = make_embedding_numeric(
                features,
                scaler=self.metadata["numeric_scaler"],
                fit=False,
            )
            categorical = encode_embeddings(
                features,
                self.metadata["embedding_maps"],
            )
            probabilities = predict_embedding(
                self.model,
                categorical,
                numeric,
                self.metadata["device"],
            )

        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, X, threshold=None):
        threshold = (
            self.metadata.get("threshold", 0.5)
            if threshold is None
            else threshold
        )
        return (
            self.predict_proba(X)[:, 1] >= threshold
        ).astype(int)


def save_artifact(artifact, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path):
    return joblib.load(path)
