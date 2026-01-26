from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from eyerunn_cluster.cognitive import extract_cognitive_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="预测单个 session 的 cluster 和 2D 坐标")
    p.add_argument("--session_dir", type=str, required=True, help="session 目录路径（包含 6 CSV + 1 JSON）")
    p.add_argument("--classifier_model", type=str, required=True, help="分类器模型路径（train_classifier.py 生成的 .joblib）")
    p.add_argument("--pca_model", type=str, required=True, help="PCA 模型路径（cluster_cognitive_data.py 生成的 pca_model.joblib）")
    p.add_argument("--features_template", type=str, default="outputs/features.csv", help="特征模板（用于对齐特征列顺序）")
    p.add_argument("--out_dir", type=str, default="outputs_predict", help="输出目录")
    p.add_argument("--plot", action="store_true", help="生成可视化图片（叠加到训练数据的散点图上）")
    p.add_argument("--training_embedding", type=str, default="outputs/embedding_2d.csv", help="训练数据的 2D 坐标（用于可视化）")
    p.add_argument("--partition_dir", type=str, default=None, help="按 cluster 分区输出目录（为空则不分区）")
    p.add_argument(
        "--partition_mode",
        type=str,
        default="copy",
        choices=["copy", "move", "list"],
        help="分区模式：copy 复制/ move 移动/ list 仅输出清单",
    )
    p.add_argument("--prob_high", type=float, default=0.8, help="高置信区间阈值（默认 0.8）")
    p.add_argument("--prob_mid", type=float, default=0.6, help="中置信区间阈值（默认 0.6）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"session 目录不存在: {session_dir}")

    # 1. 提取新 session 的特征
    print(f"[INFO] 提取特征: {session_dir}")
    feats_new = extract_cognitive_features(session_dir, unit="session")
    if len(feats_new) != 1:
        raise ValueError(f"期望提取 1 个样本，实际得到 {len(feats_new)} 个")
    sample_key = feats_new.index[0]
    print(f"[INFO] sample_key: {sample_key}")

    # 2. 对齐特征列顺序（必须和训练时一致）
    feats_template = pd.read_csv(args.features_template, index_col=0)
    feat_cols = [c for c in feats_template.columns if c != "sample_key"]
    feats_new_aligned = feats_new.reindex(columns=feat_cols, fill_value=np.nan)

    # 3. 加载分类器模型并预测 cluster
    print(f"[INFO] 加载分类器: {args.classifier_model}")
    clf_data = joblib.load(args.classifier_model)
    clf_model = clf_data["model"]
    label_encoder = clf_data["label_encoder"]

    cluster_pred = clf_model.predict(feats_new_aligned)[0]
    cluster_name = label_encoder.inverse_transform([cluster_pred])[0]

    # 获取预测概率（如果支持）
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(feats_new_aligned)[0]
        proba_dict = {label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    else:
        proba_dict = {}

    print(f"[OK] 预测 cluster: {cluster_name} (编码={cluster_pred})")

    # 4. 加载 PCA 模型并计算 2D 坐标
    print(f"[INFO] 加载 PCA: {args.pca_model}")
    pca_data = joblib.load(args.pca_model)
    pca_pipeline = pca_data["pipeline"]  # imputer + scaler
    pca_model = pca_data["pca"]

    # 用相同的预处理 pipeline
    X_processed = pca_pipeline.transform(feats_new_aligned)
    coords_2d = pca_model.transform(X_processed)[0]
    x, y = float(coords_2d[0]), float(coords_2d[1])

    print(f"[OK] 2D 坐标: ({x:.4f}, {y:.4f})")

    # 5. 保存结果
    result = {
        "sample_key": str(sample_key),
        "session_dir": str(session_dir.resolve()),
        "predicted_cluster": str(cluster_name),
        "predicted_cluster_encoded": int(cluster_pred),
        "coordinates_2d": {"x": x, "y": y},
        "probabilities": proba_dict,
    }

    result_path = out_dir / "prediction_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 结果已保存: {result_path}")

    # 6. 可选：可视化（叠加到训练数据上）
    if args.plot:
        training_emb = pd.read_csv(args.training_embedding)
        plt.figure(figsize=(8, 6))

        # 画训练数据
        for c in sorted(set(int(x) for x in training_emb["cluster"])):
            sub = training_emb[training_emb["cluster"] == c]
            label = f"cluster {c}" if c != -1 else "noise (-1)"
            plt.scatter(sub["x"], sub["y"], s=30, alpha=0.6, label=label, c=f"C{c}")

        # 画新预测的点（更大、更显眼）
        plt.scatter(
            [x],
            [y],
            s=200,
            marker="*",
            c="red",
            edgecolors="black",
            linewidths=2,
            label=f"新样本: {sample_key} (cluster {cluster_name})",
            zorder=10,
        )

        plt.title(f"预测结果 | {sample_key} → cluster {cluster_name}")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.legend(frameon=False, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = out_dir / "prediction_plot.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"[OK] 可视化已保存: {plot_path}")

    # 7. 可选：按 cluster 分区输出
    if args.partition_dir:
        partition_root = Path(args.partition_dir)
        partition_root.mkdir(parents=True, exist_ok=True)
        cluster_dir = partition_root / f"cluster_{cluster_name}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        list_path = cluster_dir / "samples.txt"
        list_path.write_text(
            (list_path.read_text(encoding="utf-8") if list_path.exists() else "") + f"{sample_key}\n",
            encoding="utf-8",
        )

        # 基于预测最大概率做粗分区
        prob_dir = None
        if proba_dict:
            max_prob = max(proba_dict.values())
            if max_prob >= args.prob_high:
                prob_dir = cluster_dir / "prob_high"
            elif max_prob >= args.prob_mid:
                prob_dir = cluster_dir / "prob_mid"
            else:
                prob_dir = cluster_dir / "prob_low"
            prob_dir.mkdir(parents=True, exist_ok=True)
            prob_list = prob_dir / "samples.txt"
            prob_list.write_text(
                (prob_list.read_text(encoding="utf-8") if prob_list.exists() else "") + f"{sample_key}\n",
                encoding="utf-8",
            )

        if args.partition_mode != "list":
            dst_root = prob_dir if prob_dir is not None else cluster_dir
            dst = dst_root / session_dir.name
            if not dst.exists():
                if args.partition_mode == "move":
                    shutil.move(str(session_dir), str(dst))
                else:
                    shutil.copytree(session_dir, dst, dirs_exist_ok=True)
        print(f"[OK] 分区输出完成: {partition_root.resolve()}")

    # 控制台输出
    print("\n" + "=" * 50)
    print(f"预测结果摘要:")
    print(f"  样本: {sample_key}")
    print(f"  预测 cluster: {cluster_name}")
    print(f"  2D 坐标: ({x:.4f}, {y:.4f})")
    if proba_dict:
        print(f"  各类别概率:")
        for k, v in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"    {k}: {v:.3f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
