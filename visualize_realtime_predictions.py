from __future__ import annotations

"""
根据 `realtime_session_monitor.py` 生成的 JSONL 日志，
可视化所有已预测 task 在二维空间中的分布（按 cluster 着色）。

用法（在项目根目录下运行）:

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    py -3.10 Model\\ET_model\\visualize_realtime_predictions.py ^
        --log_jsonl Model\\ET_model\\realtime_predictions_task_supervised.jsonl

如果你想把图保存成文件而不是/同时弹出窗口，可以加上：

    --output outputs\\realtime_2d_clusters.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取实时预测 JSONL 日志文件，绘制各个类别在二维空间中的散点分布图。"
    )
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default="Model/ET_model/realtime_predictions_task_supervised.jsonl",
        help="实时预测追加写入的 JSONL 日志文件路径（与 realtime_session_monitor.py 中的默认值一致，task 级）。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="可选，若指定则将图保存到该路径（例如 outputs/realtime_2d_clusters.png）。",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="只保存图片而不弹出窗口。",
    )
    return parser.parse_args()


def load_points_from_log(log_path: Path) -> Dict[int, List[Tuple[float, float]]]:
    """
    从 JSONL 日志中读取二维坐标和 cluster。

    日志中每行是一个 JSON，对应 append_log 写入的 payload：
        {
            "session_dir": "...",
            "sample_key": "...",
            "predicted_cluster": 0,
            "relative_load_level": ...,
            "relative_load_label": ...,
            "coordinates_2d": [x, y],
            "probabilities": {...}
        }
    """
    clusters: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    if not log_path.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            coords = obj.get("coordinates_2d")
            cluster = obj.get("predicted_cluster")

            if (
                isinstance(coords, (list, tuple))
                and len(coords) == 2
                and isinstance(cluster, int)
            ):
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                except (TypeError, ValueError):
                    continue
                clusters[cluster].append((x, y))

    return clusters


def plot_clusters(
    clusters: Dict[int, List[Tuple[float, float]]],
    *,
    title: str = "二维空间中的认知负荷聚类分布",
) -> None:
    """将不同 cluster 的点画成散点图。"""
    if not clusters:
        raise ValueError("没有从日志中读到任何有效的坐标点。")

    plt.figure(figsize=(8, 6), dpi=120)

    # 为常见的 0~3 类准备颜色和标签；超过 4 类时自动扩展颜色。
    base_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    for idx, (cluster, pts) in enumerate(sorted(clusters.items(), key=lambda x: x[0])):
        xs, ys = zip(*pts)
        color = base_colors[idx % len(base_colors)]
        plt.scatter(
            xs,
            ys,
            s=35,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
            label=f"cluster {cluster}（n={len(pts)}）",
            c=color,
        )

    plt.xlabel("PCA 维度 1", fontsize=12)
    plt.ylabel("PCA 维度 2", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.axis("equal")


def main() -> int:
    args = _parse_args()

    log_path = Path(args.log_jsonl).resolve()
    print(f"[INFO] 读取日志文件: {log_path}")

    clusters = load_points_from_log(log_path)

    total_points = sum(len(v) for v in clusters.values())
    print(f"[INFO] 总共读取到 {total_points} 个点，{len(clusters)} 个 cluster。")
    for c, pts in sorted(clusters.items(), key=lambda x: x[0]):
        print(f"  - cluster {c}: {len(pts)} 个点")

    plot_clusters(clusters)

    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"[INFO] 图像已保存到: {output_path}")

    if not args.no_show:
        print("[INFO] 弹出窗口展示散点图（关闭窗口即可结束）。")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

