from __future__ import annotations

import argparse
import json
import joblib
import shutil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# 作为包运行时（python -m Model.ET_model.cluster_cognitive_data），使用相对导入
from .eyerunn_cluster import cluster_features
from .eyerunn_cluster.cognitive import discover_sessions, extract_cognitive_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cognitive_data 格式聚类（6 CSV + 1 JSON / session）")
    p.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="包含多个 session 子目录的根目录（或直接指定某个 session 目录）；在 EyeTrace 项目中默认为项目根下的 data/ 目录",
    )
    p.add_argument("--unit", type=str, default="session", choices=["session", "task"], help="聚类单位：按会话或按任务")

    p.add_argument("--algo", type=str, default="kmeans", choices=["kmeans", "agglo", "dbscan"], help="聚类算法")
    p.add_argument("--k", type=int, default=4, help="KMeans/层次聚类簇数")
    p.add_argument("--dbscan_eps", type=float, default=0.8, help="DBSCAN eps")
    p.add_argument("--dbscan_min_samples", type=int, default=5, help="DBSCAN min_samples")
    p.add_argument("--random_state", type=int, default=42, help="随机种子")

    p.add_argument("--out_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--no_plot", action="store_true", help="不生成聚类可视化图片")
    p.add_argument("--partition_dir", type=str, default=None, help="按 cluster 分区输出目录（为空则不分区）")
    p.add_argument(
        "--partition_mode",
        type=str,
        default="copy",
        choices=["copy", "move", "list"],
        help="分区模式：copy 复制/ move 移动/ list 仅输出清单",
    )
    p.add_argument(
        "--feature_prefixes",
        type=str,
        default="fix__,blink__,trans__,task__",
        help=(
            "用于聚类的特征前缀，逗号分隔。例如 'fix__,blink__,trans__,task__' 只使用这些前缀开头的特征；"
            "留空则不做筛选，使用全部特征。"
        ),
    )
    p.add_argument(
        "--feature_weights_json",
        type=str,
        default=None,
        help=(
            "特征权重配置 JSON 文件路径（可选）。"
            "格式: {\"fix__duration__mean\": 2.0, \"trans__n\": 1.5, ...}；"
            "仅影响聚类时的距离权重，不会改变导出的 features.csv。"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 原始完整特征表（用于后续汇总与可视化）
    feats_all = extract_cognitive_features(args.data_root, unit=args.unit)
    print(f"[INFO] discovered samples: {len(feats_all)} (unit={args.unit}) from {Path(args.data_root).resolve()}")
    feats_all.to_csv(out_dir / "features.csv", index=True, encoding="utf-8-sig")

    if len(feats_all) < 2 and args.algo in ("kmeans", "agglo"):
        raise ValueError(
            f"当前只有 {len(feats_all)} 个样本，无法进行 {args.algo} 聚类。"
            "请检查 data_root 是否指向包含足够 session / task 的目录。"
        )

    # 按前缀选择用于聚类的特征子集（默认更偏向与认知负荷相关的特征）
    feats_for_cluster = feats_all.copy()
    prefix_raw = (args.feature_prefixes or "").strip()
    if prefix_raw:
        prefixes = [p.strip() for p in prefix_raw.split(",") if p.strip()]
        if prefixes:
            cols_keep = [c for c in feats_all.columns if any(c.startswith(p) for p in prefixes)]
            if not cols_keep:
                raise ValueError(
                    f"按 feature_prefixes={prefixes} 未匹配到任何特征列；"
                    f"当前可用列示例: {list(feats_all.columns)[:10]}"
                )
            feats_for_cluster = feats_all[cols_keep].copy()
            print(f"[INFO] using {len(cols_keep)} features for clustering (prefix filter: {prefixes})")

    # 读取特征权重 JSON（可选）
    feature_weights: Dict[str, float] | None = None
    if args.feature_weights_json:
        w_path = Path(args.feature_weights_json)
        if not w_path.exists():
            raise FileNotFoundError(f"feature_weights_json 文件不存在: {w_path}")
        with w_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("feature_weights_json 格式错误：应为 {\"feature_name\": weight, ...} 的对象")
        feature_weights = {}
        for k, v in raw.items():
            try:
                feature_weights[str(k)] = float(v)
            except Exception:
                continue
        print(f"[INFO] loaded {len(feature_weights)} feature weights from {w_path}")

    if args.algo in ("kmeans", "agglo") and args.k > len(feats_for_cluster) and len(feats_for_cluster) > 0:
        print(f"[WARN] k={args.k} 大于样本数 n={len(feats_for_cluster)}，已自动下调为 k={len(feats_for_cluster)}")
        args.k = int(len(feats_for_cluster))

    res = cluster_features(
        feats_for_cluster,
        algo=args.algo,
        k=args.k,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        random_state=args.random_state,
        feature_weights=feature_weights,
    )

    clusters = pd.DataFrame({"sample_key": feats_all.index, "cluster": res.labels})
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8-sig")

    emb = pd.DataFrame(
        {
            "sample_key": feats_all.index,
            "x": res.embedding_2d[:, 0],
            "y": res.embedding_2d[:, 1],
            "cluster": res.labels,
        }
    )
    emb.to_csv(out_dir / "embedding_2d.csv", index=False, encoding="utf-8-sig")

    # 保存 PCA 模型（用于后续预测新样本的 2D 坐标）
    # 注意：此处仅使用聚类时选取的特征子集，以保证维度一致
    X_processed = res.pipeline.transform(feats_for_cluster)
    pca = PCA(n_components=2, random_state=args.random_state)
    pca.fit(X_processed)
    joblib.dump({"pipeline": res.pipeline, "pca": pca}, out_dir / "pca_model.joblib")
    print(f"[OK] PCA 模型已保存: {out_dir / 'pca_model.joblib'}")

    if not args.no_plot:
        plt.figure(figsize=(7, 5))
        for c in sorted(set(int(x) for x in res.labels)):
            sub = emb[emb["cluster"] == c]
            label = f"cluster {c}" if c != -1 else "noise (-1)"
            plt.scatter(sub["x"], sub["y"], s=30, alpha=0.85, label=label)
        title = f"{args.algo} | unit={args.unit} | n={len(emb)}"
        if res.silhouette is not None:
            title += f" | silhouette={res.silhouette:.3f}"
        plt.title(title)
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "cluster_plot.png", dpi=160)
        plt.close()

    # 摘要
    # n_samples: 所有样本数（task/session）
    # n_features_used: 实际用于聚类的特征数（可能因 feature_prefixes 被筛选）
    print(f"[OK] samples: {len(feats_all)}, features_used_for_clustering: {feats_for_cluster.shape[1]}")
    uniq = sorted(set(int(x) for x in res.labels))
    print(f"[OK] clusters: {uniq}")
    if res.silhouette is not None:
        print(f"[OK] silhouette: {res.silhouette:.4f}")
    print(f"[OK] outputs written to: {out_dir.resolve()}")

    # 按 cluster 分区输出（可选）
    if args.partition_dir:
        partition_root = Path(args.partition_dir)
        partition_root.mkdir(parents=True, exist_ok=True)
        mode = args.partition_mode

        # 仅当 unit=session 时尝试复制/移动会话目录
        session_map: dict[str, Path] = {}
        if args.unit == "session":
            for sdir in discover_sessions(args.data_root):
                session_map[sdir.name] = sdir

        for sample_key, cluster_id in zip(feats.index.tolist(), res.labels.tolist()):
            cluster_dir = partition_root / f"cluster_{int(cluster_id)}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            # 每个 cluster 生成清单文件
            list_path = cluster_dir / "samples.txt"
            list_path.write_text(
                (list_path.read_text(encoding="utf-8") if list_path.exists() else "") + f"{sample_key}\n",
                encoding="utf-8",
            )

            if args.unit != "session" or mode == "list":
                continue

            src = session_map.get(str(sample_key))
            if src is None or not src.exists():
                continue

            dst = cluster_dir / src.name
            if dst.exists():
                continue

            if mode == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)

        print(f"[OK] partitioned outputs written to: {partition_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

