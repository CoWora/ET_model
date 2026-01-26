from __future__ import annotations

import argparse
import joblib
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from eyerunn_cluster import cluster_features
from eyerunn_cluster.cognitive import discover_sessions, extract_cognitive_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cognitive_data 格式聚类（6 CSV + 1 JSON / session）")
    p.add_argument(
        "--data_root",
        type=str,
        default="cognitive_data",
        help="包含多个 session 子目录的根目录（或直接指定某个 session 目录）",
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
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = extract_cognitive_features(args.data_root, unit=args.unit)
    print(f"[INFO] discovered samples: {len(feats)} (unit={args.unit}) from {Path(args.data_root).resolve()}")
    feats.to_csv(out_dir / "features.csv", index=True, encoding="utf-8-sig")

    if len(feats) < 2 and args.algo in ("kmeans", "agglo"):
        raise ValueError(f"当前只有 {len(feats)} 个样本，无法进行 {args.algo} 聚类。请检查 data_root 是否指向包含 100 个 session 的目录。")

    if args.algo in ("kmeans", "agglo") and args.k > len(feats) and len(feats) > 0:
        print(f"[WARN] k={args.k} 大于样本数 n={len(feats)}，已自动下调为 k={len(feats)}")
        args.k = int(len(feats))

    res = cluster_features(
        feats,
        algo=args.algo,
        k=args.k,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        random_state=args.random_state,
    )

    clusters = pd.DataFrame({"sample_key": feats.index, "cluster": res.labels})
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8-sig")

    emb = pd.DataFrame(
        {"sample_key": feats.index, "x": res.embedding_2d[:, 0], "y": res.embedding_2d[:, 1], "cluster": res.labels}
    )
    emb.to_csv(out_dir / "embedding_2d.csv", index=False, encoding="utf-8-sig")

    # 保存 PCA 模型（用于后续预测新样本的 2D 坐标）
    X_processed = res.pipeline.transform(feats)
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
    print(f"[OK] samples: {len(feats)}, features: {feats.shape[1]}")
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

