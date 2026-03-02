from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 兼容两种运行方式：
# 1) 作为包模块：python -m Model.ET_model.cluster_eye_tracking
# 2) 直接脚本：python Model\ET_model\cluster_eye_tracking.py
try:
    # 包内相对导入（推荐）
    from .eyerunn_cluster import (
        cluster_features,
        extract_features_per_sample,
        load_multicsv_timeseries,
    )
except ImportError:  # pragma: no cover - 仅在脚本直接运行时触发
    import sys
    from pathlib import Path as _Path

    _THIS_DIR = _Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))

    from eyerunn_cluster import (  # type: ignore[no-redef]
        cluster_features,
        extract_features_per_sample,
        load_multicsv_timeseries,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="眼动多CSV时序聚类（无监督）")
    p.add_argument("--data_dir", type=str, default="data", help="数据目录（包含 6 个 CSV + 1 个 JSON）")
    p.add_argument("--csv_glob", type=str, default="*.csv", help="CSV 匹配模式（默认 *.csv）")
    p.add_argument("--json_path", type=str, default=None, help="JSON 路径（默认自动找 data_dir 下第一个 .json）")
    p.add_argument("--id_col", type=str, default=None, help="样本 id 列名（默认自动猜测）")
    p.add_argument("--time_col", type=str, default=None, help="时间列名（默认自动猜测）")
    p.add_argument("--no_prefix", action="store_true", help="不对信号列加文件名前缀（可能会列名冲突）")

    p.add_argument("--algo", type=str, default="kmeans", choices=["kmeans", "agglo", "dbscan"], help="聚类算法")
    p.add_argument("--k", type=int, default=4, help="KMeans/层次聚类簇数")
    p.add_argument("--dbscan_eps", type=float, default=0.8, help="DBSCAN eps")
    p.add_argument("--dbscan_min_samples", type=int, default=5, help="DBSCAN min_samples")
    p.add_argument("--random_state", type=int, default=42, help="随机种子")

    p.add_argument("--out_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--no_plot", action="store_true", help="不生成聚类可视化图片")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, meta, info = load_multicsv_timeseries(
        data_dir,
        csv_glob=args.csv_glob,
        json_path=args.json_path,
        id_col=args.id_col,
        time_col=args.time_col,
        prefix_columns=not args.no_prefix,
    )

    # 保存 merge 后的原始长表（方便排查）
    merged.to_csv(out_dir / "merged_timeseries.csv", index=False, encoding="utf-8-sig")
    if meta is not None:
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    feats = extract_features_per_sample(merged, id_col="sample_id", time_col="timestamp")
    feats.to_csv(out_dir / "features.csv", index=True, encoding="utf-8-sig")

    res = cluster_features(
        feats,
        algo=args.algo,
        k=args.k,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        random_state=args.random_state,
    )

    clusters = pd.DataFrame(
        {
            "sample_id": feats.index,
            "cluster": res.labels,
        }
    )
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8-sig")

    emb = pd.DataFrame(
        {
            "sample_id": feats.index,
            "x": res.embedding_2d[:, 0],
            "y": res.embedding_2d[:, 1],
            "cluster": res.labels,
        }
    )
    emb.to_csv(out_dir / "embedding_2d.csv", index=False, encoding="utf-8-sig")

    # 简单可视化
    if not args.no_plot:
        plt.figure(figsize=(7, 5))
        for c in sorted(set(int(x) for x in res.labels)):
            sub = emb[emb["cluster"] == c]
            label = f"cluster {c}" if c != -1 else "noise (-1)"
            plt.scatter(sub["x"], sub["y"], s=30, alpha=0.85, label=label)
        title = f"{args.algo}  |  n={len(emb)}"
        if res.silhouette is not None:
            title += f"  |  silhouette={res.silhouette:.3f}"
        plt.title(title)
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "cluster_plot.png", dpi=160)
        plt.close()

    # 控制台摘要
    print(f"[OK] CSV files: {len(info.csv_paths)}")
    print(f"[OK] merged rows: {len(merged)}, cols: {len(merged.columns)}")
    print(f"[OK] samples: {len(feats)}, features: {feats.shape[1]}")
    uniq = sorted(set(int(x) for x in res.labels))
    print(f"[OK] clusters: {uniq}")
    if res.silhouette is not None:
        print(f"[OK] silhouette: {res.silhouette:.4f}")
    print(f"[OK] outputs written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

