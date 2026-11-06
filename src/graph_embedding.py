"""Inductive low-rank transaction/entity graph representation.

The SVD is fitted on the training-window incidence matrix only. Future rows
are projected using the frozen training entity vocabulary and components;
unseen entities contribute zero rather than being learned from the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class TransactionGraphSVD:
    def __init__(self, n_components: int = 12, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    @staticmethod
    def _tokens(frame: pd.DataFrame):
        tokens = []
        for col, prefix in [("user_id", "u"), ("device_id", "d"), ("ip_address", "i")]:
            tokens.extend(prefix + "::" + str(v) for v in frame[col].fillna("__MISSING__"))
        return tokens

    def _matrix(self, frame: pd.DataFrame):
        n = len(frame)
        rows, cols, values = [], [], []
        for row_idx, token in enumerate(self._tokens(frame)):
            col_idx = self.vocabulary_.get(token)
            if col_idx is not None:
                rows.append(row_idx // 3)
                cols.append(col_idx)
                values.append(1.0)
        return csr_matrix((values, (rows, cols)), shape=(n, len(self.vocabulary_)), dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        tokens = self._tokens(frame)
        self.vocabulary_ = {token: i for i, token in enumerate(sorted(set(tokens)))}
        matrix = self._matrix(frame)
        components = min(self.n_components, max(1, min(matrix.shape) - 1))
        self.svd_ = TruncatedSVD(n_components=components, random_state=self.random_state, n_iter=7)
        values = self.svd_.fit_transform(matrix)
        return self._pad(values)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "svd_"):
            raise RuntimeError("Call fit_transform on the training window first")
        return self._pad(self.svd_.transform(self._matrix(frame)))

    def _pad(self, values):
        out = np.zeros((len(values), self.n_components), dtype=np.float32)
        out[:, :values.shape[1]] = values.astype(np.float32)
        return out
