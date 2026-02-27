from __future__ import annotations

"""
基于已采集 session 目录的简单“准实时”监控脚本（方案一 - 不改采集系统）。

用法（推荐在项目根目录运行）:

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    py -3.10 Model\\ET_model\\realtime_session_monitor.py ^
        --watch_dirs Cognitive\\data\\cognitive_study data ^
        --interval 10

说明：
- 不会修改任何采集代码，只是轮询指定目录下新出现的 session 目录，
  每发现一个未处理过的目录，就调用模型进行一次预测。
- 预测使用 `predict_utils.predict_session`，输出包含 cluster、相对认知负荷等级等。
"""

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Set

from predict_utils import predict_session, PredictionResult


REQUIRED_FILES = {
    "gaze_data.csv",
    "fixations.csv",
    "aoi_transitions.csv",
    "blinks.csv",
    "tasks.csv",
    "events.csv",
    "session_meta.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="轮询监控采集数据目录，一旦出现新的 session 就自动调用模型预测认知负荷（不改采集系统）。"
    )
    parser.add_argument(
        "--watch_dirs",
        nargs="+",
        default=[
            "Cognitive/data/cognitive_study",
            "data",
        ],
        help="要监控的根目录列表（会递归查找其中的 session 子目录）。默认同时监控 Cognitive/data/cognitive_study 和 项目根下的 data。",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="轮询间隔（秒），默认 10 秒扫描一次。",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default="Model/ET_model/outputs_supervised_task/model_svm.joblib",
        help="分类器模型路径（task 级 6 类模型）。",
    )
    parser.add_argument(
        "--pca_model",
        type=str,
        default="Model/ET_model/outputs_task_cluster/pca_model.joblib",
        help="PCA 模型路径（task 级 PCA）。",
    )
    parser.add_argument(
        "--features_template",
        type=str,
        default="Model/ET_model/outputs_task_cluster/features.csv",
        help="特征模板 CSV 路径（task 级特征）。",
    )
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default="Model/ET_model/realtime_predictions_task_supervised.jsonl",
        help="将每次预测结果追加写入的 JSONL 文件路径（每行一个 JSON 对象，task 级）。",
    )
    parser.add_argument(
        "--run_once",
        action="store_true",
        help="只扫描一次并处理当前发现的 session 后退出（便于试跑/调试）。",
    )
    parser.add_argument(
        "--suppress_sklearn_warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否抑制 sklearn 常见的重复告警（例如缺失值/特征名提示）。默认开启。",
    )
    return parser.parse_args()


def is_session_dir(path: Path) -> bool:
    """判断目录是否包含一整套采集输出文件。"""
    if not path.is_dir():
        return False
    names = {p.name for p in path.iterdir() if p.is_file()}
    return REQUIRED_FILES.issubset(names)


def find_all_sessions(root_dirs: Iterable[Path]) -> Set[Path]:
    """在给定的根目录下递归查找所有 session 目录。"""
    session_paths: Set[Path] = set()
    for root in root_dirs:
        if not root.exists():
            continue
        # 只遍历一层时间戳子目录即可，避免过深递归
        for child in root.iterdir():
            if child.is_dir() and is_session_dir(child):
                session_paths.add(child.resolve())
    return session_paths


def predict_one_session(
    session_dir: Path,
    *,
    classifier_model: Path,
    pca_model: Path,
    features_template: Path,
) -> list[PredictionResult]:
    """对单个 session 目录运行一次预测。

    返回每个 task 一条的列表（task 级预测）
    """
    result = predict_session(
        session_dir,
        classifier_model=classifier_model,
        pca_model=pca_model,
        features_template=features_template,
    )
    return list(result)


def append_log(log_path: Path, session_dir: Path, results: list[PredictionResult]) -> None:
    """将预测结果以 JSON 行的形式追加写入日志文件（一条样本一行）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for r in results:
            payload = {
                "session_dir": str(session_dir),
                **asdict(r),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    args = _parse_args()

    watch_roots = [Path(p).resolve() for p in args.watch_dirs]
    classifier_model = Path(args.classifier_model).resolve()
    pca_model = Path(args.pca_model).resolve()
    features_template = Path(args.features_template).resolve()
    log_path = Path(args.log_jsonl).resolve()

    if args.suppress_sklearn_warnings:
        # 这些 warning 在“实时监控”的循环里会反复出现，信息密度低，且通常不影响结果输出。
        warnings.filterwarnings(
            "ignore",
            message=r"X has feature names, but .* was fitted without feature names",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Skipping features without any observed values: .*",
            category=UserWarning,
        )

    print("=" * 80)
    print("实时 session 监控已启动（方案一，不改采集系统）")
    print(f"- 监控目录: {', '.join(str(p) for p in watch_roots)}")
    print(f"- 扫描间隔: {args.interval} 秒")
    print(f"- 分类器模型: {classifier_model}")
    print(f"- PCA 模型: {pca_model}")
    print(f"- 特征模板: {features_template}")
    print(f"- 日志文件: {log_path}")
    print(f"- 预测粒度: task（按每个 task_id 一条）")
    print("=" * 80)

    processed: Set[Path] = set()

    try:
        while True:
            all_sessions = find_all_sessions(watch_roots)
            new_sessions = sorted(all_sessions - processed)

            if new_sessions:
                print(f"\n[INFO] 发现 {len(new_sessions)} 个新的 session 目录待预测：")
                for s in new_sessions:
                    print(f"  - {s}")

            for session_dir in new_sessions:
                print("\n" + "-" * 80)
                print(f"[INFO] 开始预测 session: {session_dir}")
                try:
                    results = predict_one_session(
                        session_dir,
                        classifier_model=classifier_model,
                        pca_model=pca_model,
                        features_template=features_template,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[ERROR] 预测失败: {session_dir}")
                    print(f"        {type(e).__name__}: {e}")
                    # 即使失败也标记为已处理，避免无限重试；需要时可以手动删除日志重新跑
                    processed.add(session_dir)
                    continue

                # 控制台输出简要结果
                print(f"[OK] 预测完成: {session_dir.name}")
                for i, result in enumerate(results, start=1):
                    task_label = result.task_id or "__all__"
                    print(f"  --- 子样本 {i} / task_id={task_label} ---")
                    print(f"    样本 ID           : {result.sample_key}")
                    print(f"    预测 cluster      : {result.predicted_cluster}")
                    print(
                        f"    相对认知负荷等级 : Level {result.relative_load_level} - {result.relative_load_label}"
                    )
                    print(
                        "    2D 坐标           : "
                        f"({result.coordinates_2d[0]:.4f}, {result.coordinates_2d[1]:.4f})"
                    )
                    if result.probabilities:
                        print("    各类别概率        :")
                        for k, v in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                            print(f"      {k}: {v:.3f}")

                append_log(log_path, session_dir, results)
                processed.add(session_dir)

            if args.run_once:
                print("\n[INFO] --run_once 已启用：本次扫描处理完成，退出。")
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，退出监控。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

