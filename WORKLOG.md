# EyeTrace · ET_model 工作日志（从训练到任务级负荷浏览）

> 说明：本日志面向“可复现、可追溯”。只记录仓库中**已落地**的脚本、参数与产物路径，不做不可验证的推断。  
> 运行环境：Windows + `py -3.10`（示例命令均以 EyeTrace 项目根目录为工作目录）。  
> 编码约定：多数 CSV 输出使用 `utf-8-sig`（便于 Excel 直接打开不乱码）。

---

## 总体目标（我们在做什么）

- **无监督聚类**：从眼动/任务相关 CSV 与 meta JSON 中提取特征，对 session 或 task 做聚类，生成 `clusters.csv`、`features.csv`、`embedding_2d.csv`、`pca_model.joblib` 等产物。
- **相对认知负荷解释层**：基于聚类后的关键特征均值，对每个 cluster 生成“相对负荷等级（1-4）+标签”，输出 `cluster_load_summary.csv` 与 `cluster_load_mapping.csv`。
- **监督训练与预测**：在已有聚类标签基础上训练分类器（SVM/XGBoost），用于对新 session 预测 cluster；并可计算 PCA 2D 坐标便于可视化。
- **任务级浏览**：对 task 粒度的聚类结果生成映射，并用离线小面板快速查看“某 session 的各任务分别属于哪个 cluster/负荷等级”。

---

## 阶段 A：无监督聚类（session/task 两种单位）

### A1. 脚本与关键参数

- 脚本：`Model/ET_model/cluster_cognitive_data.py`
- 关键参数：
  - `--data_root`：数据根目录（默认 `data/`）
  - `--unit session|task`：聚类单位
  - `--algo kmeans|agglo|dbscan`、`--k`：算法与簇数
  - `--feature_prefixes`：用于聚类的特征前缀筛选（默认 `fix__,blink__,trans__,task__`）
  - `--feature_weights_json`：特征加权配置（JSON，**作用于 StandardScaler 之后的 z-score**）

### A2. 典型命令（session 级）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\cluster_cognitive_data.py --data_root data --unit session --k 4 --out_dir Model\ET_model\outputs
```

### A3. 典型命令（task 级，带权重）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\cluster_cognitive_data.py ^
  --data_root data ^
  --unit task ^
  --k 6 ^
  --out_dir Model\ET_model\outputs_task_cluster ^
  --feature_weights_json Model\ET_model\feature_weights_task.json
```

### A4. 主要产物（out_dir 下）

- `features.csv`：样本特征（索引为 `sample_key`）
- `clusters.csv`：`sample_key, cluster`
- `embedding_2d.csv`：PCA 2D 坐标（便于可视化）
- `cluster_plot.png`：聚类散点图（可关闭）
- `pca_model.joblib`：包含 `pipeline`（imputer+scaler+weighter）与 `pca`，用于后续对齐 transform

---

## 阶段 B：cluster → 相对认知负荷等级（auto/manual）

### B1. 脚本与模式

- 脚本：`Model/ET_model/summarize_cluster_load.py`
- 输入：
  - `--features <out_dir>/features.csv`
  - `--clusters <out_dir>/clusters.csv`
- 输出（写入 `--out_dir`）：
  - `cluster_load_summary.csv`：每个 cluster 的关键特征均值 + 负荷等级/标签
  - `cluster_load_mapping.csv`：cluster → 负荷等级/标签的精简映射表（给面板/其他模块直接用）
- 生成方式：
  - `--mapping_mode auto`（默认）：对关键列做**稳健 z-score**（方差为 0 时返回 0），按加权得分排序映射到等级 1..4
  - `--mapping_mode manual`：使用脚本内置的手动映射表（当你要“强行指定某个 cluster 语义”时用）

### B2. 典型命令（对 task 聚类输出做映射）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\summarize_cluster_load.py ^
  --features Model\ET_model\outputs_task_cluster\features.csv ^
  --clusters Model\ET_model\outputs_task_cluster\clusters.csv ^
  --out_dir Model\ET_model\outputs_task_cluster ^
  --mapping_mode auto
```

---

## 阶段 C：监督训练（把聚类“固化”为可预测模型）

### C1. 训练脚本

- 脚本：`Model/ET_model/train_classifier.py`
- 输入：
  - `--features outputs/features.csv`
  - `--labels outputs/clusters.csv`
- 输出目录（默认 `outputs_supervised/`）：
  - `model_svm.joblib` 或 `model_xgboost.joblib`：保存 `model`（含预处理 pipeline）与 `label_encoder`
  - `metrics.json`：指标/说明（样本少时不做 train/test 切分）
  - `confusion_matrix.png`、`test_predictions.csv`：样本满足条件时生成
- 训练时会自动过滤 task 级样本中的 `::task=none` 段（不参与监督训练）

### C2. 典型命令（SVM）

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\train_classifier.py ^
  --features Model\ET_model\outputs\features.csv ^
  --labels Model\ET_model\outputs\clusters.csv ^
  --algo svm ^
  --out_dir Model\ET_model\outputs_supervised
```

---

## 阶段 D：预测单个 session（cluster + 2D 坐标）

### D1. 预测脚本

- 脚本：`Model/ET_model/predict_single_session.py`
- 要点：
  - `--features_template` 用于**对齐特征列顺序**
  - `--pca_model` 中的 `pipeline` 保证与聚类时相同的 transform（含加权步骤，如果当时启用了权重）

### D2. 典型命令

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\predict_single_session.py ^
  --session_dir data\20260124_140152 ^
  --classifier_model Model\ET_model\outputs_supervised\model_svm.joblib ^
  --pca_model Model\ET_model\outputs\pca_model.joblib ^
  --features_template Model\ET_model\outputs\features.csv
```

---

## 阶段 E：任务级聚类可分性问题 → “权重化聚类”落地

### E1. 问题（现象）

- 在 task 级聚类中，需要**更稳定地区分**同一 session 内的不同任务（例如 `20260216_202836` 的 `task_001 / task_002 / task_003`）。

### E2. 解决策略（做了什么）

- **特征子集**：聚类默认只使用与负荷更相关的前缀特征：`fix__ / blink__ / trans__ / task__`
- **特征加权**：在 z-score 后对列做乘权，强调更能反映“任务负荷”的维度。
  - 权重配置文件：`Model/ET_model/feature_weights_task.json`
  - 关键强调项（示例）：`task__duration__mean`、`trans__n`、`fix__duration__mean` 等
- **实现位置**：
  - 入口参数：`cluster_cognitive_data.py --feature_weights_json ...`
  - 具体实现：`Model/ET_model/eyerunn_cluster/clustering.py` 中的 `FeatureWeighter`（位于 `StandardScaler` 之后）

### E3. 结果验证点（已在当前输出中体现）

- 产物目录：`Model/ET_model/outputs_task_cluster/`
- 样例（同一 session 内任务被分到不同 cluster）：
  - `20260216_202836::task=task_001` → cluster 1
  - `20260216_202836::task=task_002` → cluster 4
  - `20260216_202836::task=task_003` → cluster 3

---

## 阶段 F：自动 cluster→负荷映射 + 离线任务浏览面板对齐

### F1. 自动映射（解决“cluster id 变化导致语义错乱/手动维护成本高”）

- `summarize_cluster_load.py` 增加 `--mapping_mode auto|manual`，默认 `auto`
- `auto` 模式做法：
  - 对关键列做稳健 z-score（方差为 0/全缺失时返回 0，避免运行警告与 NaN）
  - 按加权总分从低到高排序，线性映射到等级 1..4（并生成稳定标签）

### F2. 离线面板（读取 task 聚类结果与映射）

- 脚本：`Model/ET_model/offline_task_dashboard.py`
- 数据源固定为：
  - `Model/ET_model/outputs_task_cluster/clusters.csv`
  - `Model/ET_model/outputs_task_cluster/cluster_load_mapping.csv`
- 运行：

```bash
cd C:\Users\YNS\Desktop\EyeTrace
py -3.10 Model\ET_model\offline_task_dashboard.py
```

---

## 当前推荐工作流（最少命令集）

1) 任务级聚类（可带权重）→ `outputs_task_cluster/`  
2) 生成相对负荷映射（auto）→ 同目录生成 `cluster_load_mapping.csv`  
3) 打开离线面板快速检查（按 session 过滤查看）

---

## 备注 / 后续可选项

- 若你希望“某类任务必须固定为某个负荷等级”，可在汇总时使用：
  - `summarize_cluster_load.py --mapping_mode manual`
  - 或在 `auto` 打分权重（脚本内 `score_cols_weights`）上做领域化微调

