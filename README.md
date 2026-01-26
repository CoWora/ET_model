# 眼动时序数据分组与预测

本项目用于对眼动时序数据进行聚类分析，并可基于聚类结果训练分类器，支持对新会话进行预测与可视化。

## 功能概览
- 聚类：KMeans / 层次聚类 / DBSCAN
- 特征提取：从多 CSV 时序数据中提取统计与时序特征
- 输出可视化：PCA 2D 散点图
- 监督分类：SVM / XGBoost
- 新样本预测：输出所属 cluster 与 2D 坐标

### 1. 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 2. 放入你的数据
支持两种数据格式，选一种即可。

**格式 A（推荐）：每个 session 一个文件夹**
```
exam_0/
  cognitive_data/
    session_001/
      gaze_data.csv
      fixations.csv
      blinks.csv
      events.csv
      aoi_transitions.csv
      tasks.csv
      session_meta.json
    session_002/
      ...
```

**格式 B：所有 CSV 放同一目录**
```
exam_0/
  data/
    a.csv
    b.csv
    c.csv
    d.csv
    e.csv
    f.csv
    meta.json
```

### 3. 开始分组
格式 A：
```bash
python cluster_cognitive_data.py --data_root cognitive_data --unit session --k 4
```

可选参数：
- `--time_col` / `--id_col`: 指定时间列与样本列名
- `--csv_glob`: CSV 匹配规则（默认 *.csv）
- `--json_path`: 指定 JSON 文件

## 查看结果（输出说明）
聚类完成后会在 `outputs/` 目录生成：
- `features.csv`：每个样本的特征向量
- `clusters.csv`：聚类标签
- `embedding_2d.csv`：2D 坐标（PCA）
- `cluster_plot.png`：聚类可视化图（可选）
- `pca_model.joblib`：用于预测新样本坐标的 PCA 模型

## 查看结果
输出在 `outputs/` 目录：
- `clusters.csv`：每个样本属于哪一类  
- `embedding_2d.csv`：二维坐标  
- `cluster_plot.png`：可视化图  
- `features.csv`：特征文件（后续训练用）  

## 训练分类器（让新数据可预测）
训练步骤基于 `outputs/` 里的聚类结果：

**1）用 SVM 训练（推荐先用这个）**
```bash
python train_classifier.py --features outputs/features.csv --labels outputs/clusters.csv --algo svm --out_dir outputs_supervised
```

**2）用 XGBoost 训练（可选）**
```bash
python train_classifier.py --features outputs/features.csv --labels outputs/clusters.csv --algo xgboost --out_dir outputs_supervised_xgb
```

**训练后会生成：**
- `model_*.joblib`：训练好的模型  
- `metrics.json`：训练指标  
- `test_predictions.csv`：测试集预测（如果样本足够）  

> 样本太少时会跳过测试集评估，这是正常的。

## 预测新数据
```bash
python predict_single_session.py \
  --session_dir cognitive_data/session_001 \
  --classifier_model outputs_supervised/model_svm.joblib \
  --pca_model outputs/pca_model.joblib \
  --features_template outputs/features.csv
```

## 如果分组不满意怎么办
1) 打开 `outputs/clusters.csv`，手动改成你认为正确的类别  
2) 重新运行 `train_classifier.py`，模型会用你的修正结果再训练  
