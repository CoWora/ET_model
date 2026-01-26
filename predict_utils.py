from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from eyerunn_cluster.cognitive import extract_cognitive_features


@dataclass(frozen=True)
class PredictionResult:
    """预测结果"""
    sample_key: str
    predicted_cluster: str
    predicted_cluster_encoded: int
    coordinates_2d: tuple[float, float]
    probabilities: dict[str, float]


class SessionPredictor:
    """
    预测器类：封装模型加载和预测逻辑，可以重复使用。
    
    用法：
        predictor = SessionPredictor(
            classifier_model="outputs_supervised/model_svm.joblib",
            pca_model="outputs/pca_model.joblib",
            features_template="outputs/features.csv"
        )
        result = predictor.predict("cognitive_data/20260124_191719")
        print(result.predicted_cluster, result.coordinates_2d)
    """

    def __init__(
        self,
        classifier_model: str | Path,
        pca_model: str | Path,
        features_template: str | Path,
    ):
        """
        初始化预测器，加载模型。
        
        Args:
            classifier_model: 分类器模型路径（train_classifier.py 生成的 .joblib）
            pca_model: PCA 模型路径（cluster_cognitive_data.py 生成的 pca_model.joblib）
            features_template: 特征模板 CSV（用于对齐特征列顺序）
        """
        self.classifier_model_path = Path(classifier_model)
        self.pca_model_path = Path(pca_model)
        self.features_template_path = Path(features_template)

        if not self.classifier_model_path.exists():
            raise FileNotFoundError(f"分类器模型不存在: {self.classifier_model_path}")
        if not self.pca_model_path.exists():
            raise FileNotFoundError(f"PCA 模型不存在: {self.pca_model_path}")
        if not self.features_template_path.exists():
            raise FileNotFoundError(f"特征模板不存在: {self.features_template_path}")

        # 加载模型（延迟加载，第一次 predict 时才加载）
        self._clf_data: dict[str, Any] | None = None
        self._pca_data: dict[str, Any] | None = None
        self._feat_cols: list[str] | None = None

    def _ensure_loaded(self) -> None:
        """确保模型已加载（懒加载）"""
        if self._clf_data is None:
            self._clf_data = joblib.load(self.classifier_model_path)
        if self._pca_data is None:
            self._pca_data = joblib.load(self.pca_model_path)
        if self._feat_cols is None:
            feats_template = pd.read_csv(self.features_template_path, index_col=0)
            self._feat_cols = [c for c in feats_template.columns if c != "sample_key"]

    def predict(self, session_dir: str | Path) -> PredictionResult:
        """
        预测单个 session 的 cluster 和 2D 坐标。
        
        Args:
            session_dir: session 目录路径（包含 6 CSV + 1 JSON）
            
        Returns:
            PredictionResult: 包含预测的 cluster、2D 坐标、概率等信息
        """
        self._ensure_loaded()

        session_dir = Path(session_dir)
        if not session_dir.exists():
            raise FileNotFoundError(f"session 目录不存在: {session_dir}")

        # 1. 提取特征
        feats_new = extract_cognitive_features(session_dir, unit="session")
        if len(feats_new) != 1:
            raise ValueError(f"期望提取 1 个样本，实际得到 {len(feats_new)} 个")
        sample_key = feats_new.index[0]

        # 2. 对齐特征列顺序
        feats_new_aligned = feats_new.reindex(columns=self._feat_cols, fill_value=np.nan)

        # 3. 预测 cluster
        clf_model = self._clf_data["model"]
        label_encoder = self._clf_data["label_encoder"]

        cluster_pred = clf_model.predict(feats_new_aligned)[0]
        cluster_name = label_encoder.inverse_transform([cluster_pred])[0]

        # 获取概率
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(feats_new_aligned)[0]
            proba_dict = {label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
        else:
            proba_dict = {}

        # 4. 计算 2D 坐标
        pca_pipeline = self._pca_data["pipeline"]
        pca_model = self._pca_data["pca"]

        X_processed = pca_pipeline.transform(feats_new_aligned)
        coords_2d = pca_model.transform(X_processed)[0]
        x, y = float(coords_2d[0]), float(coords_2d[1])

        return PredictionResult(
            sample_key=str(sample_key),
            predicted_cluster=str(cluster_name),
            predicted_cluster_encoded=int(cluster_pred),
            coordinates_2d=(x, y),
            probabilities=proba_dict,
        )


def predict_session(
    session_dir: str | Path,
    *,
    classifier_model: str | Path = "outputs_supervised/model_svm.joblib",
    pca_model: str | Path = "outputs/pca_model.joblib",
    features_template: str | Path = "outputs/features.csv",
) -> PredictionResult:
    """
    便捷函数：预测单个 session（一次性使用，不需要创建 Predictor 对象）。
    
    用法：
        result = predict_session("cognitive_data/20260124_191719")
        print(f"cluster: {result.predicted_cluster}, 坐标: {result.coordinates_2d}")
    
    Args:
        session_dir: session 目录路径
        classifier_model: 分类器模型路径（默认从 outputs_supervised/ 读取）
        pca_model: PCA 模型路径（默认从 outputs/ 读取）
        features_template: 特征模板路径（默认从 outputs/ 读取）
        
    Returns:
        PredictionResult: 预测结果
    """
    predictor = SessionPredictor(
        classifier_model=classifier_model,
        pca_model=pca_model,
        features_template=features_template,
    )
    return predictor.predict(session_dir)
