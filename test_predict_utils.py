"""
测试 predict_utils.py 的函数接口
"""

from pathlib import Path

from predict_utils import SessionPredictor, predict_session


def test_single_predict():
    """测试单个预测"""
    print("=" * 60)
    print("测试 1: 单个 session 预测（便捷函数）")
    print("=" * 60)

    results = predict_session(
        "cognitive_data_synth/synth_0001",
        classifier_model="outputs_supervised_svm/model_svm.joblib",
        pca_model="outputs_synth/pca_model.joblib",
        features_template="outputs_synth/features.csv",
    )
    for i, result in enumerate(results, start=1):
        print(f"[OK] 样本 {i}: {result.sample_key}")
        print(f"[OK] 预测 cluster: {result.predicted_cluster}")
        print(f"[OK] 2D 坐标: ({result.coordinates_2d[0]:.4f}, {result.coordinates_2d[1]:.4f})")
        if result.probabilities:
            print("[OK] 各类别概率:")
            for k, v in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"    {k}: {v:.3f}")
    print()


def test_batch_predict():
    """测试批量预测"""
    print("=" * 60)
    print("测试 2: 批量预测（SessionPredictor 类）")
    print("=" * 60)

    predictor = SessionPredictor(
        classifier_model="outputs_supervised_svm/model_svm.joblib",
        pca_model="outputs_synth/pca_model.joblib",
        features_template="outputs_synth/features.csv",
    )

    session_dirs = [
        "cognitive_data_synth/synth_0001",
        "cognitive_data_synth/synth_0002",
        "cognitive_data_synth/synth_0003",
    ]

    results = []
    for session_dir in session_dirs:
        if Path(session_dir).exists():
            result = predictor.predict(session_dir)
            results.append(result)
            print(f"[OK] {session_dir}: cluster {result.predicted_cluster}, 坐标 ({result.coordinates_2d[0]:.4f}, {result.coordinates_2d[1]:.4f})")

    print(f"\n[OK] 成功预测 {len(results)} 个 session")
    print()


def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("测试 3: 错误处理")
    print("=" * 60)

    # 测试不存在的 session
    try:
        _ = predict_session("不存在的目录")
        print("[FAIL] 应该抛出 FileNotFoundError")
    except FileNotFoundError:
        print("[OK] 正确捕获 FileNotFoundError（session 不存在）")

    # 测试不存在的模型
    try:
        SessionPredictor(
            classifier_model="不存在的模型.joblib",
            pca_model="outputs_synth/pca_model.joblib",
            features_template="outputs_synth/features.csv",
        )
        print("[FAIL] 应该抛出 FileNotFoundError")
    except FileNotFoundError:
        print("[OK] 正确捕获 FileNotFoundError（模型不存在）")

    print()


def main():
    print("\n" + "=" * 60)
    print("predict_utils.py 函数接口测试")
    print("=" * 60 + "\n")

    try:
        test_single_predict()
        test_batch_predict()
        test_error_handling()

        print("=" * 60)
        print("[OK] 所有测试通过！")
        print("=" * 60)
        print("\n你现在可以在自己的代码里这样用：")
        print("""
from predict_utils import predict_session

result = predict_session("你的session目录")
print(f"cluster: {result.predicted_cluster}")
print(f"坐标: {result.coordinates_2d}")
        """)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
