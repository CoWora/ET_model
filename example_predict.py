"""
示例：如何使用 predict_utils 预测单个 session

运行方式：
    python example_predict.py
"""

from pathlib import Path

from predict_utils import SessionPredictor, predict_session


def main():
    # ===== 方式 1：使用便捷函数（最简单） =====
    print("=" * 60)
    print("方式 1：使用便捷函数 predict_session()")
    print("=" * 60)

    # 假设你的模型文件在默认位置
    results = predict_session(
        "cognitive_data_synth/synth_0001",
        classifier_model="outputs_supervised_svm/model_svm.joblib",
        pca_model="outputs_synth/pca_model.joblib",
        features_template="outputs_synth/features.csv",
    )

    for i, result in enumerate(results, start=1):
        print(f"样本 {i}: {result.sample_key}")
        print(f"预测 cluster: {result.predicted_cluster}")
        print(f"2D 坐标: ({result.coordinates_2d[0]:.4f}, {result.coordinates_2d[1]:.4f})")
        if result.probabilities:
            print("各类别概率:")
            for k, v in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {k}: {v:.3f}")

    # ===== 方式 2：使用 Predictor 类（适合批量预测） =====
    print("\n" + "=" * 60)
    print("方式 2：使用 SessionPredictor 类（适合批量预测）")
    print("=" * 60)

    # 创建预测器（模型只加载一次）
    predictor = SessionPredictor(
        classifier_model="outputs_supervised_svm/model_svm.joblib",
        pca_model="outputs_synth/pca_model.joblib",
        features_template="outputs_synth/features.csv",
    )

    # 批量预测多个 session
    session_dirs = [
        "cognitive_data_synth/synth_0001",
        "cognitive_data_synth/synth_0002",
        "cognitive_data_synth/synth_0003",
    ]

    all_results = []
    for session_dir in session_dirs:
        if Path(session_dir).exists():
            results = predictor.predict(session_dir)
            all_results.extend(results)
            print(f"\n{session_dir}:")
            for r in results:
                print(f"  sample_key: {r.sample_key} | cluster: {r.predicted_cluster}, 坐标: ({r.coordinates_2d[0]:.4f}, {r.coordinates_2d[1]:.4f})")

    # ===== 方式 3：在你的代码里直接调用 =====
    print("\n" + "=" * 60)
    print("方式 3：在你的代码里直接调用")
    print("=" * 60)
    print("""
# 在你的 Python 脚本里：
from predict_utils import predict_session

# 预测一个 session（返回该 session 内所有 task 的结果列表）
results = predict_session("你的数据/cognitive_data/20260124_191719")

for r in results:
    cluster = r.predicted_cluster  # 例如: "0"~"5"
    x, y = r.coordinates_2d       # 例如: (1.234, -0.567)
    proba = r.probabilities        # 例如: {"0": 0.8, "1": 0.15, "2": 0.05}

    print(f"task {r.task_id} 被分到 cluster {cluster}")
    print(f"在 PCA 图上的位置: ({x:.2f}, {y:.2f})")
""")


if __name__ == "__main__":
    main()
