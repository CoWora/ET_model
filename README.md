# 眼动时序数据分组与预测

## 这个项目能做什么
- 多组眼动数据自动分组  
- 输入一组新数据，返回最可能的类别和概率  
- 输出结果文件与可视化图片  

## 快速上手（3 步）

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

格式 B：
```bash
python cluster_eye_tracking.py --data_dir data --k 4
```

## 查看结果
输出在 `outputs/` 目录：
- `clusters.csv`：每个样本属于哪一类  
- `embedding_2d.csv`：二维坐标  
- `cluster_plot.png`：可视化图  
- `features.csv`：特征文件（后续训练用）  

## 训练分类器（让新数据可预测）
```bash
python train_classifier.py --features outputs/features.csv --labels outputs/clusters.csv --algo svm --out_dir outputs_supervised
```

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
