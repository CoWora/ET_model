from __future__ import annotations

"""
一个简单的实时预测窗口面板：

用法（在项目根目录运行）::

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    py -3.10 Model\\ET_model\\realtime_dashboard.py

说明：
- 该脚本会定时读取 ``Model/ET_model/realtime_predictions_task_supervised.jsonl``，
  将最新的若干条 **task 级** 预测结果显示在一个简易窗口中。
- 建议与 ``realtime_session_monitor.py`` 同时运行。
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import ttk, messagebox


LOG_PATH = Path("Model/ET_model/realtime_predictions_task_supervised.jsonl")
REFRESH_INTERVAL_SEC = 2.0
MAX_ROWS = 50


@dataclass
class DisplayRecord:
    idx: int
    session_name: str
    cluster: str
    level: str
    coords: str
    proba_top: str


def _format_record(raw: dict, idx: int) -> DisplayRecord:
    session_dir = raw.get("session_dir", "")
    session_name = Path(session_dir).name or session_dir

    cluster = str(raw.get("predicted_cluster", ""))
    level_num = raw.get("relative_load_level", "")
    level_label = raw.get("relative_load_label", "")
    level = f"L{level_num} {level_label}" if level_num != "" else ""

    coords = ""
    coords_raw = raw.get("coordinates_2d")
    if isinstance(coords_raw, (list, tuple)) and len(coords_raw) == 2:
        try:
            coords = f"({float(coords_raw[0]):.3f}, {float(coords_raw[1]):.3f})"
        except Exception:  # noqa: BLE001
            coords = str(coords_raw)

    proba_top = ""
    probs = raw.get("probabilities") or {}
    if isinstance(probs, dict) and probs:
        try:
            k, v = max(probs.items(), key=lambda x: x[1])
            proba_top = f"{k}: {float(v):.3f}"
        except Exception:  # noqa: BLE001
            proba_top = ""

    return DisplayRecord(
        idx=idx,
        session_name=session_name,
        cluster=cluster,
        level=level,
        coords=coords,
        proba_top=proba_top,
    )


def load_latest_records(limit: int = MAX_ROWS) -> List[DisplayRecord]:
    if not LOG_PATH.exists():
        return []

    lines: list[str] = []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []

    records: List[DisplayRecord] = []
    start_idx = max(0, len(lines) - limit)
    for idx, line in enumerate(lines[start_idx:], start=start_idx + 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(_format_record(data, idx))

    return records


class RealtimeDashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EyeTrace 实时预测面板")
        self.root.geometry("900x400")

        self._create_widgets()

        self._refresh_lock = threading.Lock()
        self._running = True
        self._schedule_refresh()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_widgets(self) -> None:
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        lbl = ttk.Label(
            top_frame,
            text=f"数据源: {LOG_PATH} （每 {REFRESH_INTERVAL_SEC:.0f} 秒自动刷新）",
        )
        lbl.pack(side=tk.LEFT)

        btn_refresh = ttk.Button(top_frame, text="立即刷新", command=self.manual_refresh)
        btn_refresh.pack(side=tk.RIGHT, padx=4)

        btn_help = ttk.Button(top_frame, text="帮助", command=self.show_help)
        btn_help.pack(side=tk.RIGHT, padx=4)

        columns = ("idx", "session", "cluster", "level", "coords", "proba_top")
        tree = ttk.Treeview(
            self.root,
            columns=columns,
            show="headings",
            height=15,
        )
        tree.heading("idx", text="#")
        tree.heading("session", text="Session")
        tree.heading("cluster", text="Cluster")
        tree.heading("level", text="相对负荷")
        tree.heading("coords", text="2D 坐标")
        tree.heading("proba_top", text="最高概率")

        tree.column("idx", width=40, anchor=tk.CENTER)
        tree.column("session", width=200, anchor=tk.W)
        tree.column("cluster", width=80, anchor=tk.CENTER)
        tree.column("level", width=180, anchor=tk.W)
        tree.column("coords", width=180, anchor=tk.CENTER)
        tree.column("proba_top", width=160, anchor=tk.W)

        vsb = ttk.Scrollbar(self.root, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        vsb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))

        self.tree = tree

    def _schedule_refresh(self) -> None:
        if not self._running:
            return
        self.root.after(int(REFRESH_INTERVAL_SEC * 1000), self._do_refresh)

    def _do_refresh(self) -> None:
        if not self._running:
            return

        def work() -> None:
            with self._refresh_lock:
                records = load_latest_records()

            def update_ui() -> None:
                # 清空并重新填充
                for item in self.tree.get_children():
                    self.tree.delete(item)

                for rec in records:
                    self.tree.insert(
                        "",
                        tk.END,
                        values=(
                            rec.idx,
                            rec.session_name,
                            rec.cluster,
                            rec.level,
                            rec.coords,
                            rec.proba_top,
                        ),
                    )

            self.root.after(0, update_ui)

        threading.Thread(target=work, daemon=True).start()
        self._schedule_refresh()

    def manual_refresh(self) -> None:
        # 手动刷新时立刻触发一次后台加载
        self._do_refresh()

    def show_help(self) -> None:
        msg = (
            "EyeTrace 实时预测面板\n\n"
            "用法：\n"
            "1. 先运行数据采集程序，确保 session 目录不断生成；\n"
            "2. 再运行 realtime_session_monitor.py，让模型对新 session 做预测；\n"
            "3. 最后运行本窗口脚本 realtime_dashboard.py，查看最新 task 级预测结果。\n\n"
            "窗口会每隔数秒自动重新读取 realtime_predictions_task_supervised.jsonl 并更新列表。"
        )
        messagebox.showinfo("帮助", msg)

    def on_close(self) -> None:
        self._running = False
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    _ = RealtimeDashboardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

