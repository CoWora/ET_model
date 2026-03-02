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
# 在 EyeTrace 项目根目录下，直接使用 data/ 作为根目录（包含多个 session 子目录）
python Model/ET_model/cluster_cognitive_data.py --data_root data --unit session --k 4 --out_dir Model/ET_model/outputs
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
- `cluster_load_summary.csv`：每个 cluster 的关键特征均值 + 相对认知负荷等级
- `cluster_load_mapping.csv`：cluster → 相对认知负荷等级的精简映射表

## 训练分类器（让新数据可预测）
训练步骤基于 `outputs/` 里的聚类结果：

**1）用 SVM 训练（推荐先用这个）**
```bash
python Model/ET_model/train_classifier.py --features outputs/features.csv --labels outputs/clusters.csv --algo svm --out_dir outputs_supervised
```

**2）用 XGBoost 训练（可选）**
```bash
python Model/ET_model/train_classifier.py --features outputs/features.csv --labels outputs/clusters.csv --algo xgboost --out_dir outputs_supervised_xgb
```

**训练后会生成：**
- `model_*.joblib`：训练好的模型  
- `metrics.json`：训练指标  
- `test_predictions.csv`：测试集预测（如果样本足够）  

> 样本太少时会跳过测试集评估，这是正常的。

## 预测新数据（离线单 session）
```bash
python Model/ET_model/predict_single_session.py \
  --session_dir data/20260124_140152 \
  --classifier_model Model/ET_model/outputs_supervised/model_svm.joblib \
  --pca_model Model/ET_model/outputs/pca_model.joblib \
  --features_template Model/ET_model/outputs/features.csv
```
预测结果 `PredictionResult` 中同时包含：
- `predicted_cluster`：所属聚类
- `coordinates_2d`：二维坐标
- `relative_load_level` / `relative_load_label`：基于当前聚类分析得到的相对认知负荷等级

## 已训练好的模型位置（示例）

- `outputs_supervised/`：基于 **session 级聚类** 训练的监督模型  
  - `model_svm.joblib` / `model_xgboost.joblib`
  - `metrics.json` 等训练指标
- `outputs_supervised_task/`：基于 **任务级聚类** 训练的监督模型（推荐用于实时任务级预测）  
  - `model_svm.joblib`：当前默认使用的任务级 SVM 模型  
  - `metrics.json`：当前样本量下的训练指标（主要用于 sanity check）

> 这两个目录中的模型文件都已经纳入 Git 管理，克隆仓库后可直接加载使用（无需重新训练），具体训练过程与配置在 `WORKLOG.md` 的阶段 C/G 中有完整记录。

## 实时任务级预测与总控脚本（核心入口）

> 详细设计、试跑记录和字段说明，见 `WORKLOG.md` 中的“阶段 G：任务级监督模型 + 实时预测”部分。下面只给一个“最少上手说明”。

### 总控脚本：`realtime_session_monitor.py`

- 功能：  
  - 持续监听 `data/xxxxxx_realtime/` 目录下由采集端写入的 AOI/眼动数据  
  - 实时聚合“当前任务”的特征，并调用**任务级监督模型**预测其所属 cluster + 相对负荷等级  
  - 将每一次预测追加写入 `realtime_predictions_task_supervised.jsonl`
- 依赖的已训练模型与模板：  
  - `outputs_task_cluster/features.csv`：任务级特征模板（对齐列名/顺序）  
  - `outputs_task_cluster/pca_model.joblib`：任务级 PCA + 预处理 pipeline  
  - `outputs_supervised_task/model_svm.joblib`：任务级监督 SVM 模型

### 常见报错：特征维度不匹配（例如 191 vs 185）

如果你看到类似报错：

- `X has 191 features, but SimpleImputer is expecting 185 features as input.`

含义是：**预测时喂给模型的特征列数/顺序** 与 **该模型训练时拟合(SimpleImputer/Pipeline fit)的特征列数/顺序** 不一致（常见原因：`features.csv` 模板增删了列，但模型还是旧的）。

解决方式（推荐按优先级）：

- **重新训练模型**：用同一套 `features.csv` + `clusters.csv` 重新训练，产出新的 `model_*.joblib`。
- **确保模板与模型来自同一次输出**：`--features_template`（或 `--task_features_template`）应指向训练该模型时那份 `features.csv`。
- **使用最新代码**：新版本会在 `model_*.joblib` 中保存训练用 `feature_columns`，预测时优先按模型列名对齐，并在模板不一致时打印明确警告。

### 典型运行方式（示例）

在 EyeTrace 项目根目录下：

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\realtime_session_monitor.py ^
  --data_root data ^
  --task_classifier Model\ET_model\outputs_supervised_task\model_svm.joblib ^
  --task_pca_model Model\ET_model\outputs_task_cluster\pca_model.joblib ^
  --task_features_template Model\ET_model\outputs_task_cluster\features.csv
```

- 采集端负责持续往 `data/20260227_233556_realtime/` 这类目录写入 AOI/眼动 CSV；  
- 总控脚本会周期性读取最新数据、更新特征，并将预测结果写入：  
  - `Model/ET_model/realtime_predictions_task_supervised.jsonl`

## 如果分组不满意怎么办
1) 打开 `outputs/clusters.csv`，手动改成你认为正确的类别  
2) 重新运行 `Model/ET_model/train_classifier.py`，模型会用你的修正结果再训练  
