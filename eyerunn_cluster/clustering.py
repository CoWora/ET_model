from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


Algo = Literal["kmeans", "agglo", "dbscan"]


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    embedding_2d: np.ndarray
    silhouette: float | None
    model: object
    pipeline: Pipeline


def cluster_features(
    features: pd.DataFrame,
    *,
    algo: Algo = "kmeans",
    k: int = 4,
    dbscan_eps: float = 0.8,
    dbscan_min_samples: int = 5,
    random_state: int = 42,
) -> ClusterResult:
    """
    对特征表进行聚类。features 每行一个样本。
    """
    if features.empty:
        raise ValueError("features 为空")

    X = features.to_numpy(dtype="float64", copy=True)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    Xs = pipe.fit_transform(X)

    if algo == "kmeans":
        model = KMeans(n_clusters=int(k), n_init="auto", random_state=random_state)
        labels = model.fit_predict(Xs)
    elif algo == "agglo":
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        labels = model.fit_predict(Xs)
    elif algo == "dbscan":
        model = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
        labels = model.fit_predict(Xs)
    else:
        raise ValueError(f"未知 algo: {algo}")

    # 2D embedding（PCA，便于快速可视化）
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(Xs)

    # silhouette（DBSCAN 有 -1 噪声类；簇数太少/太多时会报错）
    sil: float | None = None
    try:
        uniq = sorted(set(int(x) for x in labels))
        n_clusters = len([u for u in uniq if u != -1])
        if n_clusters >= 2:
            sil = float(silhouette_score(Xs, labels))
    except Exception:
        sil = None

    return ClusterResult(labels=labels, embedding_2d=emb, silhouette=sil, model=model, pipeline=pipe)

