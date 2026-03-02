from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from predict_utils import PredictionResult, predict_session


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="预测单个 session 内所有 task 的 cluster / 2D 坐标"
    )
    p.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="session 目录路径（包含 6 CSV + 1 JSON）",
    )
    p.add_argument(
        "--classifier_model",
        type=str,
        required=True,
        help="分类器模型路径（train_classifier.py 生成的 .joblib）",
    )
    p.add_argument(
        "--pca_model",
        type=str,
        required=True,
        help="PCA 模型路径（cluster_cognitive_data.py 生成的 pca_model.joblib）",
    )
    p.add_argument(
        "--features_template",
        type=str,
        default="outputs/features.csv",
        help="特征模板（用于对齐特征列顺序）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs_predict",
        help="输出目录（JSON / JSONL）",
    )
    return p.parse_args()


def _result_to_dict(r: PredictionResult, session_dir: Path) -> dict:
    """将 PredictionResult 转为可写入 JSON 的 dict，并补充 session_dir 字段。"""
    base = asdict(r)
    base["session_dir"] = str(session_dir.resolve())
    # coordinates_2d 默认是 (x, y)，转成显式字段便于前端使用
    x, y = base.pop("coordinates_2d")
    base["coordinates_2d"] = {"x": float(x), "y": float(y)}
    return base


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    args = _parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"session 目录不存在: {session_dir}")

    print("=" * 80)
    print(f"[INFO] 开始预测 session: {session_dir}")
    print(f"- 预测粒度: task（按每个 task_id 一条）")
    print(f"- 分类器模型: {args.classifier_model}")
    print(f"- PCA 模型: {args.pca_model}")
    print(f"- 特征模板: {args.features_template}")
    print("=" * 80)

    result = predict_session(
        session_dir,
        classifier_model=args.classifier_model,
        pca_model=args.pca_model,
        features_template=args.features_template,
    )

    # 统一为列表，便于后续处理（predict_session 已经返回 list[PredictionResult]）
    results = list(result)

    # 保存到磁盘
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # task 级：多条结果写 JSON 与 JSONL，便于后续分析
    rows = [_result_to_dict(r, session_dir) for r in results]
    list_path = out_dir / "prediction_results_task.json"
    jsonl_path = out_dir / "prediction_results_task.jsonl"
    _write_json(list_path, rows)          # 整体列表
    _write_jsonl(jsonl_path, rows)        # 一行一条，兼容实时报文
    print(f"[OK] task 级结果已保存: {list_path}")
    print(f"[OK] task 级 JSONL 已保存: {jsonl_path}")

    # 控制台简要摘要
    print("\n预测结果摘要")
    print("-" * 80)
    for i, r in enumerate(results, start=1):
        task_label = r.task_id or "__all__"
        print(f"[{i}] sample_key={r.sample_key} | session_id={r.session_id} | task_id={task_label}")
        print(f"    cluster: {r.predicted_cluster} (encoded={r.predicted_cluster_encoded})")
        print(
            f"    相对负荷: Level {r.relative_load_level} - {r.relative_load_label}"
        )
        x, y = r.coordinates_2d
        print(f"    2D 坐标: ({x:.4f}, {y:.4f})")
        if r.probabilities:
            probs_sorted = sorted(r.probabilities.items(), key=lambda x: x[1], reverse=True)
            print("    各类别概率:")
            for k, v in probs_sorted:
                print(f"      {k}: {v:.3f}")
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
