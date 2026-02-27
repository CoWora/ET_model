from __future__ import annotations

"""
根据无监督聚类结果，为每个 cluster 计算代表性统计并附上“相对认知负荷等级”。

默认读取:
- features: outputs/features.csv
- clusters: outputs/clusters.csv

输出:
- outputs/cluster_load_summary.csv  每个 cluster 的关键特征均值 + 负荷等级
- outputs/cluster_load_mapping.csv  简单的 cluster→负荷等级映射表
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 兼容旧版本：仍保留手动映射（当你想“强行指定某个 cluster 的语义”时可用）
MANUAL_CLUSTER_LOAD_MAPPING: dict[str, dict[str, object]] = {
    "0": {"level": 2, "label": "低负荷 / 轻量任务型"},
    "1": {"level": 4, "label": "高负荷 / 持续专注解题型"},
    "2": {"level": 2, "label": "低负荷 / 轻量任务型"},
    "3": {"level": 1, "label": "极低负荷 / 轻松浏览型"},
}


def _labels_for_level(level: int) -> str:
    # 简单、稳定的等级标签（避免每次聚类后 cluster id 变化导致含义错乱）
    return {
        1: "极低负荷 / 轻松浏览型",
        2: "低负荷 / 轻量任务型",
        3: "中高负荷 / 信息整合型",
        4: "高负荷 / 持续专注解题型",
    }.get(int(level), "")


def _robust_zscore(x: pd.Series) -> pd.Series:
    """
    在 cluster 维度上做 z-score，遇到全相等/方差为 0 时返回 0。
    """
    x = pd.to_numeric(x, errors="coerce")
    if x.isna().all():
        return pd.Series(np.zeros(len(x), dtype="float64"), index=x.index)
    mu = float(np.nanmean(x.to_numpy(dtype="float64")))
    sigma = float(np.nanstd(x.to_numpy(dtype="float64")))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype="float64"), index=x.index)
    return (x - mu) / sigma


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据聚类结果汇总 cluster 统计并附加相对认知负荷等级")
    p.add_argument(
        "--features",
        type=str,
        default="outputs/features.csv",
        help="特征文件路径（含 sample_key 列）",
    )
    p.add_argument(
        "--clusters",
        type=str,
        default="outputs/clusters.csv",
        help="聚类标签文件路径（含 sample_key, cluster 列）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="输出目录（默认 outputs）",
    )
    p.add_argument(
        "--mapping_mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="cluster→相对负荷等级的生成方式：auto 根据关键特征自动打分排序；manual 使用脚本内置的手动映射表",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    features_path = Path(args.features)
    clusters_path = Path(args.clusters)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(f"features 文件不存在: {features_path}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters 文件不存在: {clusters_path}")

    feats = pd.read_csv(features_path)
    labels = pd.read_csv(clusters_path)

    if "sample_key" not in feats.columns:
        raise KeyError("features 文件缺少 sample_key 列")
    if "sample_key" not in labels.columns or "cluster" not in labels.columns:
        raise KeyError("clusters 文件必须包含 sample_key 和 cluster 列")

    df = feats.merge(labels, on="sample_key", how="inner")
    if df.empty:
        raise ValueError("features 与 clusters 按 sample_key 合并后为空，请检查输入文件是否对应同一批次数据")

    # 针对项目汇报里用到的代表性指标做聚合
    key_cols = [
        "fix__duration__mean",
        "fix__duration__std",
        "fix__aoi_region__n_unique",
        "fix__aoi_region__entropy",
        "fix__aoi_region__top1_prop",
        "blink__n",
        "trans__n",
        "trans__same_frac",
        "task__n",
        "task__duration__mean",
        "task__difficulty__mean",
        "task__subjective_difficulty__mean",
        "task__subjective_effort__mean",
    ]
    # 只保留实际存在的列，避免不同版本特征列略有变动时报错
    key_cols = [c for c in key_cols if c in df.columns]

    summary = df.groupby("cluster")[key_cols].mean(numeric_only=True).reset_index()

    # --- 自动打分：根据关键指标（跨 cluster 的相对均值）推断“相对认知负荷” ---
    # 说明：这不是绝对负荷，仅是“同一批次聚类结果内部”的相对排序。
    score_cols_weights: dict[str, float] = {
        # 时间/工作量
        "task__duration__mean": 3.0,
        "task__duration__std": 1.0,
        # 眼动切换强度/搜索强度
        "trans__n": 2.5,
        "trans__same_frac": -1.0,  # 更“重复”通常意味着更少探索（倾向低负荷）
        "trans__from_aoi__entropy": 1.0,
        "trans__to_aoi__entropy": 1.0,
        # 注视稳定性与信息分布
        "fix__duration__mean": 1.5,
        "fix__duration__std": 1.0,
        "fix__aoi_region__entropy": 1.2,
        "fix__aoi_region__n_unique": 1.0,
        "blink__n": 0.5,
        # 主观量表（若存在）
        "task__subjective_effort__mean": 1.5,
        "task__subjective_difficulty__mean": 1.0,
        "task__difficulty__mean": 0.8,
    }
    score_cols_weights = {c: w for c, w in score_cols_weights.items() if c in summary.columns}

    if args.mapping_mode == "manual":
        def _level_for_cluster(c: int | str) -> int | None:
            info = MANUAL_CLUSTER_LOAD_MAPPING.get(str(c))
            return int(info["level"]) if info is not None else None

        def _label_for_cluster(c: int | str) -> str | None:
            info = MANUAL_CLUSTER_LOAD_MAPPING.get(str(c))
            return str(info["label"]) if info is not None else None

        summary["relative_load_level"] = summary["cluster"].apply(_level_for_cluster)
        summary["relative_load_label"] = summary["cluster"].apply(_label_for_cluster)
    else:
        # 计算每个 cluster 的负荷分数（z-score 后加权求和）
        if not score_cols_weights:
            # 没有任何可用的打分列，就退化为 None
            summary["relative_load_level"] = None
            summary["relative_load_label"] = None
        else:
            score = pd.Series(np.zeros(len(summary), dtype="float64"), index=summary.index)
            for c, w in score_cols_weights.items():
                score = score + _robust_zscore(summary[c]) * float(w)
            summary["_relative_load_score"] = score

            # 按 score 排序（低→高），映射到 1..4
            order = summary["_relative_load_score"].rank(method="average", ascending=True)
            n = float(len(summary))
            if n <= 1:
                levels = pd.Series([2], index=summary.index, dtype="int64")
            else:
                # 线性映射到 [1,4]，再四舍五入，确保覆盖面稳定
                frac = (order - 1.0) / (n - 1.0)
                levels = (1.0 + 3.0 * frac).round().astype("int64")
                levels = levels.clip(1, 4)

            summary["relative_load_level"] = levels
            summary["relative_load_label"] = summary["relative_load_level"].apply(_labels_for_level)

    summary_path = out_dir / "cluster_load_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 额外输出一个简单的映射表，方便前端或其他模块单独引用
    unique_clusters = sorted(df["cluster"].unique())
    mapping_rows: list[dict[str, object]] = []
    for c in unique_clusters:
        info = MANUAL_CLUSTER_LOAD_MAPPING.get(str(c)) if args.mapping_mode == "manual" else None
        # auto 模式下以 summary 的计算结果为准
        row_summary = summary[summary["cluster"] == c]
        auto_level = None
        auto_label = None
        if not row_summary.empty:
            auto_level = row_summary.iloc[0].get("relative_load_level")
            auto_label = row_summary.iloc[0].get("relative_load_label")
        mapping_rows.append(
            {
                "cluster": int(c),
                "relative_load_level": int(info["level"]) if info is not None else (int(auto_level) if pd.notna(auto_level) else None),
                "relative_load_label": str(info["label"]) if info is not None else (str(auto_label) if pd.notna(auto_label) else None),
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = out_dir / "cluster_load_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    print(f"[OK] cluster 统计与负荷等级已写入: {summary_path.resolve()}")
    print(f"[OK] cluster→负荷等级映射已写入: {mapping_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

