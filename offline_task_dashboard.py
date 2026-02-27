from __future__ import annotations

"""
离线任务认知负荷浏览小面板。

用法（在项目根目录运行）::

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    py -3.10 Model\\ET_model\\offline_task_dashboard.py

说明：
- 读取 ``Model/ET_model/outputs_task_cluster/clusters.csv`` 中每个 session / task 的 cluster（按 task 级特征聚类的结果），
  再结合 ``Model/ET_model/outputs_task_cluster/cluster_load_mapping.csv`` 显示对应的相对认知负荷等级。
- 方便快速浏览：每个任务属于哪个 session、对应哪个 cluster、负荷等级与文字标签。
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox


# 使用按 task 聚类后的结果文件
# 注意：这里选择的是新的按 task 聚类输出目录 `outputs_task_cluster`，
# 这样可以与最新的聚类脚本保持一致，避免和旧版 `outputs_task` 冲突。
CLUSTERS_PATH = Path("Model/ET_model/outputs_task_cluster/clusters.csv")
MAPPING_PATH = Path("Model/ET_model/outputs_task_cluster/cluster_load_mapping.csv")


@dataclass
class TaskRecord:
    session: str
    task_id: str
    cluster: str
    level: str
    label: str


def _parse_sample_key(sample_key: str) -> Tuple[str, str]:
    """
    将 sample_key 解析为 (session, task_id).

    例如:
    - 20260124_140152::task=none      -> ("20260124_140152", "none")
    - 20260124_140152::task=task_001  -> ("20260124_140152", "task_001")
    """
    session = sample_key
    task_id = ""

    if "::" in sample_key:
        session, rest = sample_key.split("::", 1)
        if rest.startswith("task="):
            task_id = rest.split("=", 1)[1]
        else:
            task_id = rest
    return session, task_id


def load_cluster_mapping(path: Path = MAPPING_PATH) -> Dict[str, Dict[str, str]]:
    """
    读取 cluster→负荷等级映射表。

    返回:
        {cluster_str: {"level": "3", "label": "中高负荷 / 多源信息整合型"}, ...}
    """
    mapping: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = str(row.get("cluster", "")).strip()
            if c == "":
                continue
            level = str(row.get("relative_load_level", "")).strip()
            label = str(row.get("relative_load_label", "")).strip()
            mapping[c] = {
                "level": level,
                "label": label,
            }
    return mapping


def load_task_records(
    clusters_path: Path = CLUSTERS_PATH,
    mapping_path: Path = MAPPING_PATH,
) -> List[TaskRecord]:
    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters 文件不存在: {clusters_path}")

    mapping = load_cluster_mapping(mapping_path)

    records: List[TaskRecord] = []
    with clusters_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "sample_key" not in reader.fieldnames or "cluster" not in reader.fieldnames:
            raise KeyError("clusters.csv 必须包含 sample_key 和 cluster 列")

        for row in reader:
            sample_key = str(row.get("sample_key", "")).strip()
            cluster_str = str(row.get("cluster", "")).strip()
            if sample_key == "" or cluster_str == "":
                continue

            session, task_id = _parse_sample_key(sample_key)
            info = mapping.get(cluster_str, {})
            level_num = info.get("level", "")
            label = info.get("label", "")
            level_display = f"L{level_num}" if level_num != "" else ""

            records.append(
                TaskRecord(
                    session=session,
                    task_id=task_id,
                    cluster=cluster_str,
                    level=level_display,
                    label=label,
                )
            )

    # 按 session / task 排一下方便查看
    records.sort(key=lambda r: (r.session, r.task_id))
    return records


class OfflineTaskDashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EyeTrace 任务级认知负荷浏览")
        self.root.geometry("1000x500")

        self._all_records: List[TaskRecord] = []
        self._create_widgets()
        self._load_data()

    def _create_widgets(self) -> None:
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        lbl_path = ttk.Label(
            top_frame,
            text=f"数据源: {CLUSTERS_PATH} + {MAPPING_PATH}",
        )
        lbl_path.pack(side=tk.LEFT)

        btn_reload = ttk.Button(top_frame, text="重新加载", command=self._load_data)
        btn_reload.pack(side=tk.RIGHT, padx=4)

        btn_help = ttk.Button(top_frame, text="帮助", command=self._show_help)
        btn_help.pack(side=tk.RIGHT, padx=4)

        # 过滤区：按 session 筛选
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        ttk.Label(filter_frame, text="按 Session 筛选:").pack(side=tk.LEFT)
        self.session_var = tk.StringVar(value="")
        self.session_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.session_var,
            state="readonly",
            width=30,
        )
        self.session_combo.pack(side=tk.LEFT, padx=4)
        self.session_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_view())

        btn_clear = ttk.Button(filter_frame, text="清除筛选", command=self._clear_filter)
        btn_clear.pack(side=tk.LEFT, padx=4)

        # 表格
        columns = ("session", "task_id", "cluster", "level", "label")
        tree = ttk.Treeview(
            self.root,
            columns=columns,
            show="headings",
            height=18,
        )
        tree.heading("session", text="Session")
        tree.heading("task_id", text="Task")
        tree.heading("cluster", text="Cluster")
        tree.heading("level", text="等级")
        tree.heading("label", text="负荷标签")

        tree.column("session", width=160, anchor=tk.W)
        tree.column("task_id", width=120, anchor=tk.W)
        tree.column("cluster", width=80, anchor=tk.CENTER)
        tree.column("level", width=80, anchor=tk.CENTER)
        tree.column("label", width=520, anchor=tk.W)

        vsb = ttk.Scrollbar(self.root, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        vsb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))

        self.tree = tree

    def _load_data(self) -> None:
        try:
            records = load_task_records()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("加载失败", f"读取数据时出错：\n{e}")
            return

        self._all_records = records

        # 更新 session 下拉框
        sessions = sorted({r.session for r in records})
        self.session_combo["values"] = ["（全部）"] + sessions
        if not self.session_var.get():
            self.session_var.set("（全部）")

        self._refresh_view()

    def _clear_filter(self) -> None:
        self.session_var.set("（全部）")
        self._refresh_view()

    def _filtered_records(self) -> List[TaskRecord]:
        selected = self.session_var.get()
        if selected in ("", "（全部）"):
            return self._all_records
        return [r for r in self._all_records if r.session == selected]

    def _refresh_view(self) -> None:
        records = self._filtered_records()

        # 清空
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 填充
        for rec in records:
            self.tree.insert(
                "",
                tk.END,
                values=(rec.session, rec.task_id, rec.cluster, rec.level, rec.label),
            )

    def _show_help(self) -> None:
        msg = (
            "EyeTrace 任务级认知负荷浏览\n\n"
            "数据来源：\n"
            f"- {CLUSTERS_PATH}（按 task 级特征聚类的结果）\n"
            f"- {MAPPING_PATH}\n\n"
            "提示：\n"
            "1. 如果你重新跑了按 task 聚类和负荷汇总脚本，点击“重新加载”即可刷新视图；\n"
            "2. 可以通过上方下拉框按某个 session 只看对应任务的负荷情况。"
        )
        messagebox.showinfo("帮助", msg)


def main() -> int:
    root = tk.Tk()
    _ = OfflineTaskDashboardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

