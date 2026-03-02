from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


Algo = Literal["svm", "xgboost"]


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    keys: pd.Series


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 outputs/features.csv + outputs/clusters.csv 做监督分类（SVM/XGBoost）")

    p.add_argument("--features", type=str, default="outputs/features.csv", help="特征文件（features.csv）")
    p.add_argument("--labels", type=str, default="outputs/clusters.csv", help="标签文件（clusters.csv）")
    p.add_argument("--key_col", type=str, default="sample_key", help="主键列名（默认 sample_key）")
    p.add_argument("--label_col", type=str, default="cluster", help="标签列名（默认 cluster）")

    p.add_argument("--algo", type=str, default="svm", choices=["svm", "xgboost"], help="分类器")
    p.add_argument("--test_size", type=float, default=0.25, help="测试集比例（样本太少会自动禁用）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--out_dir", type=str, default="outputs_supervised", help="输出目录")
    p.add_argument("--no_plot", action="store_true", help="不输出混淆矩阵图片")
    return p.parse_args()


def _load_dataset(features_path: Path, labels_path: Path, key_col: str, label_col: str) -> Dataset:
    fdf = pd.read_csv(features_path)
    ldf = pd.read_csv(labels_path)

    if key_col not in fdf.columns:
        raise ValueError(f"features 缺少主键列 {key_col}，实际列：{list(fdf.columns)[:10]}...")
    if key_col not in ldf.columns or label_col not in ldf.columns:
        raise ValueError(f"labels 需要包含 {key_col} 与 {label_col}，实际列：{list(ldf.columns)}")

    merged = fdf.merge(ldf[[key_col, label_col]], on=key_col, how="inner")

    # 若使用 task 级样本，丢弃 sample_key 中带有 "::task=none" 的占位段
    # 这些是“无任务/休息”段，不应参与监督训练
    if key_col in merged.columns:
        key_str = merged[key_col].astype(str)
        mask_valid = ~key_str.str.contains("::task=none")
        merged = merged[mask_valid]
    if merged.empty:
        raise ValueError("features 与 labels 合并后为空：请检查 key_col 是否一致，或路径是否指向同一批输出")

    keys = merged[key_col].astype(str)
    y = merged[label_col]
    X = merged.drop(columns=[key_col, label_col])

    # 强制数值化（非数值列转 NaN，后续由 imputer 处理）
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return Dataset(X=X, y=y, keys=keys)


def _can_stratify(y: np.ndarray) -> bool:
    # stratify 需要每个类别至少 2 个样本
    vals, counts = np.unique(y, return_counts=True)
    return bool(len(vals) >= 2 and np.all(counts >= 2))


def _build_model(algo: Algo, seed: int, n_classes: int) -> Pipeline:
    if algo == "svm":
        # SVM 对尺度敏感：需要 StandardScaler
        clf = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    # algo == "xgboost"
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 xgboost：请先运行 `python -m pip install xgboost` 或 `pip install -r requirements.txt`"
        ) from e

    objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        objective=objective,
        eval_metric="logloss" if n_classes <= 2 else "mlogloss",
        tree_method="hist",
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _step(idx: int, total: int, msg: str) -> None:
    print(f"[STEP {idx}/{total}] {msg}")


def main() -> int:
    total_steps = 7
    step = 1
    args = _parse_args()
    _step(step, total_steps, "解析参数并准备输出目录")
    step += 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _step(step, total_steps, "读取特征与标签数据")
    step += 1
    ds = _load_dataset(Path(args.features), Path(args.labels), args.key_col, args.label_col)
    print(f"[INFO] dataset: n={len(ds.X)}, d={ds.X.shape[1]}, labels={ds.y.nunique()}")

    # 标签编码（保证模型输入是 0..K-1）
    _step(step, total_steps, "编码标签并构建模型")
    step += 1
    le = LabelEncoder()
    y_enc = le.fit_transform(ds.y.astype(str))
    label_names = [str(x) for x in le.classes_]
    n_classes = len(label_names)

    model = _build_model(args.algo, args.seed, n_classes)

    # 样本太少时，不做 train/test 切分，直接全量训练并导出模型
    _step(step, total_steps, "训练模型")
    step += 1
    do_split = len(ds.X) >= 8 and _can_stratify(y_enc) and 0.0 < args.test_size < 1.0
    if do_split:
        X_train, X_test, y_train, y_test, k_train, k_test = train_test_split(
            ds.X,
            y_enc,
            ds.keys.to_numpy(),
            test_size=float(args.test_size),
            random_state=int(args.seed),
            stratify=y_enc,
        )
        model.fit(X_train, y_train)
        _step(step, total_steps, "评估模型并生成指标")
        step += 1
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "algo": args.algo,
                    "n_samples": int(len(ds.X)),
                    "n_features": int(ds.X.shape[1]),
                    "labels": label_names,
                    "test_accuracy": acc,
                    "classification_report": report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        pred_df = pd.DataFrame(
            {
                args.key_col: k_test,
                "y_true": le.inverse_transform(y_test),
                "y_pred": le.inverse_transform(y_pred),
            }
        )
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

        if not args.no_plot:
            _save_confusion_matrix(cm, labels=label_names, path=out_dir / "confusion_matrix.png", title=f"{args.algo} CM")

        print(f"[OK] test_accuracy={acc:.4f}  (test_size={args.test_size})")
    else:
        model.fit(ds.X, y_enc)
        _step(step, total_steps, "保存训练指标（无测试集）")
        step += 1
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "algo": args.algo,
                    "n_samples": int(len(ds.X)),
                    "n_features": int(ds.X.shape[1]),
                    "labels": label_names,
                    "note": "样本较少或类别计数不足，未进行 train/test 切分；已使用全量数据训练模型。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("[WARN] 样本较少/类别计数不足，未做测试集评估；已全量训练。")

    # 保存模型（包含预处理 pipeline）
    _step(step, total_steps, "保存模型文件")
    model_path = out_dir / f"model_{args.algo}.joblib"
    # 关键：保存训练时的特征列名，便于预测阶段严格对齐
    # 否则当 features.csv 增删列后，会出现：
    #   X has N features, but SimpleImputer is expecting M features as input.
    feature_columns = [str(c) for c in ds.X.columns]
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "key_col": args.key_col,
            "feature_columns": feature_columns,
            "n_features": int(ds.X.shape[1]),
        },
        model_path,
    )
    print(f"[OK] model saved: {model_path.resolve()}")
    print(f"[OK] outputs written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

