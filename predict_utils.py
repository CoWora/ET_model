from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# 兼容两种运行方式：
# 1) 作为包模块：python -m Model.ET_model.xxx
# 2) 直接脚本：python Model\ET_model\xxx.py
try:
    # 包内相对导入（推荐）
    from .eyerunn_cluster.cognitive import extract_cognitive_features
except ImportError:  # pragma: no cover - 仅在脚本直接运行时触发
    import sys
    from pathlib import Path as _Path

    _THIS_DIR = _Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))

    from eyerunn_cluster.cognitive import extract_cognitive_features  # type: ignore[no-redef]


# 兼容：某些 joblib 产物在序列化时记录了 `Model.ET_model...` 的模块路径。
# 当脚本以 `py Model\\ET_model\\xxx.py` 方式运行时，sys.path 往往只包含 `Model/ET_model`，
# 不包含项目根目录，导致反序列化时报 `No module named 'Model'`。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# 兜底映射：当找不到 `cluster_load_mapping.csv` 时使用
# 注意：这里已经对齐到 **task 级 6 类模型** 的默认语义，
# 与 `outputs_task_cluster/cluster_load_mapping.csv` 一致。
_FALLBACK_CLUSTER_LOAD_MAPPING: dict[str, tuple[int, str]] = {
    "0": (2, "低负荷 / 轻量任务型"),
    "1": (2, "低负荷 / 轻量任务型"),
    "2": (3, "中高负荷 / 信息整合型"),
    "3": (4, "高负荷 / 持续专注解题型"),
    "4": (3, "中高负荷 / 信息整合型"),
    "5": (1, "极低负荷 / 轻松浏览型"),
}

DEFAULT_LOAD_LEVEL = 0
DEFAULT_LOAD_LABEL = "未知负荷"


def _load_cluster_load_mapping_csv(path: Path) -> dict[str, tuple[int, str]]:
    """
    读取 summarize_cluster_load.py 输出的 cluster→负荷映射表。

    期望列:
      - cluster
      - relative_load_level
      - relative_load_label
    """
    mapping: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = str(row.get("cluster", "")).strip()
            if c == "":
                continue
            try:
                level = int(str(row.get("relative_load_level", "")).strip())
            except Exception:
                level = DEFAULT_LOAD_LEVEL
            label = str(row.get("relative_load_label", "")).strip()
            if label == "":
                label = DEFAULT_LOAD_LABEL
            mapping[c] = (level, label)
    return mapping


def get_relative_load_for_cluster(
    cluster: str | int,
    *,
    mapping: dict[str, tuple[int, str]] | None = None,
) -> tuple[int, str]:
    """
    根据 cluster 编号返回相对认知负荷等级与文字描述。

    优先使用外部 mapping（通常来自 `cluster_load_mapping.csv`），找不到则回退到内置映射。
    """
    m = mapping or _FALLBACK_CLUSTER_LOAD_MAPPING
    return m.get(str(cluster), (DEFAULT_LOAD_LEVEL, DEFAULT_LOAD_LABEL))


@dataclass(frozen=True)
class PredictionResult:
    """单条 **task 级** 预测结果。"""

    # 原始样本键（格式通常为 "session_id::task=xxx"）
    sample_key: str
    # 解析出的 session / task 维度，方便前端展示
    session_id: str
    task_id: str | None

    predicted_cluster: str
    predicted_cluster_encoded: int
    coordinates_2d: tuple[float, float]
    probabilities: dict[str, float]
    # 新增：相对认知负荷等级及其说明
    relative_load_level: int
    relative_load_label: str


class SessionPredictor:
    """
    预测器类：封装模型加载和预测逻辑，可以重复使用。
    
        用法：
            predictor = SessionPredictor(
                classifier_model=\"Model/ET_model/outputs_supervised/model_svm.joblib\",
                pca_model=\"Model/ET_model/outputs/pca_model.joblib\",
                features_template=\"Model/ET_model/outputs/features.csv\",
            )
            result = predictor.predict(\"data/20260124_140152\")
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
        self._cluster_load_mapping: dict[str, tuple[int, str]] | None = None

    def _ensure_loaded(self) -> None:
        """确保模型已加载（懒加载）"""
        if self._clf_data is None:
            self._clf_data = joblib.load(self.classifier_model_path)
        if self._pca_data is None:
            self._pca_data = joblib.load(self.pca_model_path)
        if self._feat_cols is None:
            # 优先使用模型内保存的训练列名（train_classifier.py 会写入 feature_columns）
            # 这样即使外部 features.csv / features_template 演进增删列，也不会导致维度不匹配报错。
            model_cols = None
            try:
                model_cols = self._clf_data.get("feature_columns")  # type: ignore[union-attr]
            except Exception:
                model_cols = None

            if isinstance(model_cols, list) and model_cols:
                self._feat_cols = [str(c) for c in model_cols]
            else:
                feats_template = pd.read_csv(self.features_template_path, index_col=0)
                self._feat_cols = [c for c in feats_template.columns if c != "sample_key"]

            # 额外校验：如果模板列数与模型期望不一致，给出更明确的提示（但仍按模型列对齐）
            try:
                feats_template = pd.read_csv(self.features_template_path, index_col=0)
                template_cols = [c for c in feats_template.columns if c != "sample_key"]
                if self._feat_cols and len(template_cols) != len(self._feat_cols):
                    print(
                        "[WARN] features_template 与分类器模型的训练特征维度不一致："
                        f"template={len(template_cols)} vs model={len(self._feat_cols)}。"
                        "将按模型训练列名对齐（建议重新训练模型或使用训练时的同一 features.csv 作为模板）。"
                    )
            except Exception:
                # 模板读取失败时不影响预测（仍按模型列或后续逻辑）
                pass
        if self._cluster_load_mapping is None:
            # 默认：从 features_template 同目录下自动读取映射表
            mapping_path = self.features_template_path.parent / "cluster_load_mapping.csv"
            loaded = _load_cluster_load_mapping_csv(mapping_path)
            self._cluster_load_mapping = loaded if loaded else _FALLBACK_CLUSTER_LOAD_MAPPING

    def _predict_from_features(
        self,
        feats_row: pd.DataFrame,
    ) -> tuple[str, PredictionResult]:
        """从单行特征 DataFrame 进行一次预测，并构造 PredictionResult。"""
        assert len(feats_row) == 1
        sample_key = str(feats_row.index[0])

        # 解析 sample_key → session_id / task_id
        if "::task=" in sample_key:
            session_id, task_part = sample_key.split("::task=", 1)
            task_id: str | None = task_part or None
        else:
            session_id = sample_key
            task_id = None

        # 2. 对齐特征列顺序
        feats_new_aligned = feats_row.reindex(columns=self._feat_cols, fill_value=np.nan)

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

        # NOTE:
        # `pca_model.joblib` 可能是用“特征子集”训练的（例如 task 级聚类默认只用 fix__/blink__/trans__/task__ 前缀），
        # 这会导致 pca_pipeline.n_features_in_ < 全量特征列数。
        # 为了避免直接报错，这里在维度不一致时尝试按默认前缀筛出同等维度的子集；仍不匹配则返回 NaN 坐标。
        feats_for_pca = feats_new_aligned
        expected_n = getattr(pca_pipeline, "n_features_in_", None)
        if expected_n is not None and feats_new_aligned.shape[1] != int(expected_n):
            default_prefixes = ("fix__", "blink__", "trans__", "task__")
            cols_subset = [c for c in feats_new_aligned.columns if c.startswith(default_prefixes)]
            if len(cols_subset) == int(expected_n):
                feats_for_pca = feats_new_aligned[cols_subset]
            else:
                feats_for_pca = None

        if feats_for_pca is None:
            x = float("nan")
            y = float("nan")
        else:
            X_processed = pca_pipeline.transform(feats_for_pca)
            coords_2d = pca_model.transform(X_processed)[0]
            x, y = float(coords_2d[0]), float(coords_2d[1])

        # 5. 相对认知负荷等级
        load_level, load_label = get_relative_load_for_cluster(
            cluster_name,
            mapping=self._cluster_load_mapping,
        )

        result = PredictionResult(
            sample_key=sample_key,
            session_id=session_id,
            task_id=task_id,
            predicted_cluster=str(cluster_name),
            predicted_cluster_encoded=int(cluster_pred),
            coordinates_2d=(x, y),
            probabilities=proba_dict,
            relative_load_level=load_level,
            relative_load_label=load_label,
        )
        return sample_key, result

    def predict(
        self,
        session_dir: str | Path,
    ) -> list[PredictionResult]:
        """
        预测**单个 session 内所有 task** 的 cluster 和 2D 坐标。

        Args:
            session_dir: session 目录路径（包含 6 CSV + 1 JSON）

        Returns:
            list[PredictionResult]：按 task 列表返回，每个 task 一条
        """
        self._ensure_loaded()

        session_dir = Path(session_dir)
        if not session_dir.exists():
            raise FileNotFoundError(f"session 目录不存在: {session_dir}")

        # 1. 提取 task 级特征
        feats_new = extract_cognitive_features(session_dir, unit="task")
        if feats_new.empty:
            raise ValueError("未能从该 session 提取到任何特征")

        # 每一行是一个 task 样本
        results: list[PredictionResult] = []
        for idx in feats_new.index:
            # 这里用 DataFrame 保持与 _predict_from_features 的接口一致
            feats_row = feats_new.loc[[idx]]
            _, r = self._predict_from_features(feats_row)
            results.append(r)
        return results


def predict_session(
    session_dir: str | Path,
    *,
    classifier_model: str | Path = "outputs_supervised_task/model_svm.joblib",
    pca_model: str | Path = "outputs_task_cluster/pca_model.joblib",
    features_template: str | Path = "outputs_task_cluster/features.csv",
) -> list[PredictionResult]:
    """
    便捷函数：预测单个 session 内 **所有 task** 的 cluster/负荷等级/2D 坐标。
    
    用法：
        result = predict_session(\"data/20260124_140152\")
        print(f\"cluster: {result.predicted_cluster}, 坐标: {result.coordinates_2d}\")
    
    Args:
        session_dir: session 目录路径
        classifier_model: 分类器模型路径（默认使用 task 级 6 类模型：outputs_supervised_task/model_svm.joblib）
        pca_model: PCA 模型路径（默认使用 task 级 PCA：outputs_task_cluster/pca_model.joblib）
        features_template: 特征模板路径（默认使用 task 级特征模板：outputs_task_cluster/features.csv）
        
    Returns:
        list[PredictionResult]: 每个 task 一条预测结果
    """
    predictor = SessionPredictor(
        classifier_model=classifier_model,
        pca_model=pca_model,
        features_template=features_template,
    )
    return predictor.predict(session_dir)
