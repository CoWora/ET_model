from __future__ import annotations

"""
认知负荷总控 + 可视化窗口（不再显示眼动小预览窗，只展示认知负荷）。

用法（推荐在项目根目录运行）:

    cd C:\\Users\\YNS\\Desktop\\EyeTrace
    # 1) 正常启动原来的采集脚本（带眼动小窗）
    py -3.10 Cognitive\\cognitive-load-tracker\\cognitive_study\\aoi_collector_v3_2.py

    # 2) 另外开一个终端，启动本总控可视化脚本：
    py -3.10 Model\\ET_model\\cognitive_load_dashboard.py

说明：
- 本脚本会自动启动 `realtime_session_monitor.py`，对采集产生的 session 目录做预测，
  并从 `realtime_predictions_task_supervised.jsonl` 里读取结果做可视化。
- 界面只展示“相对认知负荷等级”的随时间变化，不再显示眼睛注视点的小窗口。
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "Model" / "ET_model" / "realtime_predictions_task_supervised.jsonl"
MONITOR_SCRIPT = PROJECT_ROOT / "Model" / "ET_model" / "realtime_session_monitor.py"


@dataclass
class LoadRecord:
    index: int
    session_dir: str
    task_id: str
    relative_load_level: int
    relative_load_label: str
    sample_key: str


def start_monitor_process() -> subprocess.Popen:
    """
    启动后台的 realtime_session_monitor 进程，用于不断扫描新产生的 session 目录并写入 JSONL。
    """
    if not MONITOR_SCRIPT.exists():
        print(f"[WARN] 找不到监控脚本: {MONITOR_SCRIPT}")
        return None  # type: ignore[return-value]

    cmd = [
        sys.executable,
        str(MONITOR_SCRIPT),
        "--watch_dirs",
        "Cognitive/data/cognitive_study",
        "data",
        "--interval",
        "30",  # 每 30 秒扫描一次
    ]
    print(f"[INFO] 启动实时监控进程: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc


def load_recent_records(max_points: int = 50) -> List[LoadRecord]:
    """
    读取 JSONL 日志中最近若干条预测结果。
    """
    if not LOG_PATH.exists():
        return []

    records: list[LoadRecord] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                rec = LoadRecord(
                    index=idx,
                    session_dir=obj.get("session_dir", ""),
                    task_id=str(obj.get("task_id") or "__all__"),
                    relative_load_level=int(obj.get("relative_load_level", -1)),
                    relative_load_label=str(obj.get("relative_load_label", "")),
                    sample_key=str(obj.get("sample_key", "")),
                )
            except Exception:
                continue
            records.append(rec)

    return records[-max_points:]


def color_for_level(level: int) -> str:
    """
    不同认知负荷等级对应不同颜色（可根据实际等级数量调整）。
    """
    palette = {
        0: "#4caf50",  # 低
        1: "#8bc34a",
        2: "#ffc107",
        3: "#ff9800",
        4: "#f44336",  # 高
        5: "#9c27b0",
    }
    return palette.get(level, "#9e9e9e")


def run_dashboard(poll_interval: float = 2.0) -> None:
    """
    打开一个简单的 matplotlib 窗口，实时刷新最近若干条预测的认知负荷等级。
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.canvas.manager.set_window_title("Cognitive Load Dashboard (Task-level)")

    text_box = fig.text(
        0.01,
        0.98,
        "",
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
    )

    while True:
        records = load_recent_records(max_points=40)

        ax.clear()
        ax.set_facecolor("#202124")
        fig.patch.set_facecolor("#202124")

        if not records:
            ax.text(
                0.5,
                0.5,
                "等待预测结果输出...\n请确认采集脚本和监控脚本已运行。",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
            )
            ax.set_xticks([])
            ax.set_yticks(range(0, 6))
            ax.set_ylim(-0.5, 5.5)
            plt.pause(poll_interval)
            continue

        xs = list(range(len(records)))
        ys = [r.relative_load_level for r in records]
        colors = [color_for_level(lvl) for lvl in ys]

        bars = ax.bar(xs, ys, color=colors)

        # x 轴标签用 "session短ID#task_id"
        labels = [f"{Path(r.session_dir).name}#{r.task_id}" for r in records]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

        ax.set_ylabel("Relative Load Level", color="white")
        ax.set_ylim(-0.5, max(ys) + 0.5)
        ax.grid(axis="y", color="#404040", linestyle="--", linewidth=0.5)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        # 在每个柱子上方标记等级
        for bar, rec in zip(bars, records):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"L{rec.relative_load_level}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
            )

        latest = records[-1]
        text_box.set_text(
            f"最新预测 - Session: {Path(latest.session_dir).name}  "
            f"Task: {latest.task_id}  "
            f"Level {latest.relative_load_level} - {latest.relative_load_label}"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.pause(poll_interval)


def main() -> int:
    print("=" * 80)
    print("认知负荷总控可视化窗口已启动")
    print(f"- 项目根目录: {PROJECT_ROOT}")
    print(f"- 预测日志:   {LOG_PATH}")
    print("=" * 80)

    monitor_proc = start_monitor_process()
    if monitor_proc is None:
        print("[WARN] 未能自动启动监控进程，请手动运行 realtime_session_monitor.py")

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，准备退出。")
    finally:
        if monitor_proc is not None:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=5)
            except Exception:
                pass
            print("[INFO] 已结束监控进程。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

