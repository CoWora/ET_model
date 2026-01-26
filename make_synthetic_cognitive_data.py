from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AOI_REGIONS = [
    ("F_CODE_EDITOR", "Code Editor"),
    ("F_LUOGU", "Luogu"),
    ("F_TERMINAL", "Terminal"),
    ("F_BROWSER", "Browser"),
    ("F_OTHER", "Other"),
]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _simulate_session(
    *,
    rng: np.random.Generator,
    session_id: str,
    start_time: float,
    pattern: int,
    n_tasks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    生成一套 cognitive_data 格式文件内容：
    - gaze_data.csv
    - fixations.csv
    - blinks.csv
    - events.csv
    - aoi_transitions.csv
    - tasks.csv
    - session_meta.json
    """
    # 3 个“潜在簇”模式，让聚类更容易出结果
    base_freq = [0.6, 1.8, 0.25][pattern % 3]
    noise = [0.06, 0.12, 0.04][pattern % 3]
    aoi_bias = [
        {"F_CODE_EDITOR": 0.55, "F_LUOGU": 0.10, "F_TERMINAL": 0.20, "F_BROWSER": 0.10, "F_OTHER": 0.05},
        {"F_CODE_EDITOR": 0.25, "F_LUOGU": 0.45, "F_TERMINAL": 0.10, "F_BROWSER": 0.15, "F_OTHER": 0.05},
        {"F_CODE_EDITOR": 0.70, "F_LUOGU": 0.05, "F_TERMINAL": 0.10, "F_BROWSER": 0.10, "F_OTHER": 0.05},
    ][pattern % 3]

    # 任务表
    task_rows = []
    task_ids: list[str] = []
    t_cursor = start_time
    for j in range(n_tasks):
        tid = f"task_{session_id}_{j+1:02d}"
        task_ids.append(tid)
        dur = float(rng.uniform(25, 55))  # seconds
        t0 = t_cursor
        t1 = t_cursor + dur
        t_cursor = t1 + float(rng.uniform(3, 8))  # gap
        task_rows.append(
            {
                "task_id": tid,
                "problem_id": int(rng.integers(1000, 9999)),
                "difficulty": int(rng.integers(1, 6)),
                "start_time": t0,
                "end_time": t1,
                "duration": dur,
                "result": rng.choice(["AC", "WA", "TLE", "RE", "NA"], p=[0.55, 0.15, 0.05, 0.05, 0.20]),
                "subjective_difficulty": float(rng.uniform(1, 5)),
                "subjective_effort": float(rng.uniform(1, 5)),
            }
        )
    tasks = pd.DataFrame(task_rows)

    # gaze_data：按任务拼接时间轴
    gaze_rows = []
    fixation_rows = []
    blink_rows = []
    transition_rows = []
    event_rows = []

    event_rows.append({"timestamp": start_time, "type": "SESSION_START", "task_id": "none", "description": ""})

    fixation_id = 0
    blink_id = 0
    last_aoi: str | None = None

    for tr in task_rows:
        tid = tr["task_id"]
        t0 = float(tr["start_time"])
        t1 = float(tr["end_time"])
        event_rows.append({"timestamp": t0, "type": "TASK_START", "task_id": tid, "description": ""})

        # 采样：~30-90Hz（不规则）
        n = int(rng.integers(900, 2200))
        dt = rng.uniform(1 / 95, 1 / 28, size=n)
        ts = t0 + np.cumsum(dt)
        ts = ts[ts <= t1]
        if ts.size < 10:
            continue

        # gaze 模拟
        t_rel = ts - ts[0]
        gx = 0.50 + 0.12 * np.sin(2 * np.pi * base_freq * t_rel + pattern) + rng.normal(0, noise, size=ts.size)
        gy = 0.50 + 0.10 * np.cos(2 * np.pi * base_freq * t_rel + 0.3 * pattern) + rng.normal(0, noise, size=ts.size)
        gx = np.clip(gx, 0, 1)
        gy = np.clip(gy, 0, 1)
        sx = np.clip((gx * 2560).round().astype(int), 0, 2559)
        sy = np.clip((gy * 1440).round().astype(int), 0, 1439)
        yaw = rng.normal(0, 0.5, size=ts.size) + 0.2 * np.sin(2 * np.pi * 0.15 * t_rel)
        pitch = rng.normal(0, 0.5, size=ts.size) + 0.2 * np.cos(2 * np.pi * 0.17 * t_rel)

        # AOI 序列（按偏好抽样 + 少量马尔可夫惯性）
        regions = list(aoi_bias.keys())
        probs = np.array([aoi_bias[r] for r in regions], dtype="float64")
        probs = probs / probs.sum()
        aoi_seq = []
        for k in range(ts.size):
            if k > 0 and rng.random() < 0.88:
                aoi_seq.append(aoi_seq[-1])
            else:
                aoi_seq.append(str(rng.choice(regions, p=probs)))
        aoi_name_map = {r: n for r, n in AOI_REGIONS}
        aoi_names = [aoi_name_map.get(r, "Other") for r in aoi_seq]
        is_luogu = np.array([r == "F_LUOGU" for r in aoi_seq], dtype=bool)

        # fixation：简单分段（每段 8~25 个点）
        is_fix = np.zeros(ts.size, dtype=bool)
        seg = 0
        i = 0
        while i < ts.size:
            seg_len = int(rng.integers(8, 26))
            j = min(ts.size, i + seg_len)
            # 70% 段为 fixation
            if rng.random() < 0.70:
                fixation_id += 1
                is_fix[i:j] = True
                fx0 = float(ts[i])
                fx1 = float(ts[j - 1])
                fixation_rows.append(
                    {
                        "fixation_id": fixation_id,
                        "start_time": fx0,
                        "end_time": fx1,
                        "duration": float(max(0.0, fx1 - fx0)),
                        "center_x": float(np.mean(sx[i:j])),
                        "center_y": float(np.mean(sy[i:j])),
                        "aoi_region": aoi_seq[i],
                        "task_id": tid,
                    }
                )
            i = j
            seg += 1

        fixation_id_series = np.where(is_fix, fixation_id, 0).astype(int)

        # transitions：当 AOI 变化时记录
        for k in range(ts.size):
            cur = aoi_seq[k]
            if last_aoi is None:
                last_aoi = cur
                continue
            if cur != last_aoi:
                transition_rows.append({"timestamp": float(ts[k]), "from_aoi": last_aoi, "to_aoi": cur, "task_id": tid})
                last_aoi = cur

        # blinks：按 rate 生成
        blink_rate = [0.06, 0.10, 0.05][pattern % 3]  # per second
        expected = blink_rate * float(t1 - t0)
        n_blinks = int(rng.poisson(lam=max(0.0, expected)))
        if n_blinks > 0:
            bt = rng.uniform(t0, t1, size=n_blinks)
            bt.sort()
            for btt in bt:
                blink_id += 1
                blink_rows.append(
                    {
                        "timestamp": float(btt),
                        "blink_id": blink_id,
                        "ear": float(rng.uniform(0.10, 0.35)),
                        "task_id": tid,
                    }
                )

        for k in range(ts.size):
            gaze_rows.append(
                {
                    "timestamp": float(ts[k]),
                    "gaze_x": float(gx[k]),
                    "gaze_y": float(gy[k]),
                    "screen_x": int(sx[k]),
                    "screen_y": int(sy[k]),
                    "yaw": float(yaw[k]),
                    "pitch": float(pitch[k]),
                    "aoi_region": aoi_seq[k],
                    "aoi_name": aoi_names[k],
                    "is_luogu": bool(is_luogu[k]),
                    "is_fixation": bool(is_fix[k]),
                    "fixation_id": int(fixation_id_series[k]),
                    "task_id": tid,
                }
            )

        event_rows.append({"timestamp": t1, "type": "TASK_END", "task_id": tid, "description": ""})

    gaze = pd.DataFrame(gaze_rows)
    fixations = pd.DataFrame(fixation_rows, columns=["fixation_id", "start_time", "end_time", "duration", "center_x", "center_y", "aoi_region", "task_id"])
    blinks = pd.DataFrame(blink_rows, columns=["timestamp", "blink_id", "ear", "task_id"])
    events = pd.DataFrame(event_rows, columns=["timestamp", "type", "task_id", "description"])
    transitions = pd.DataFrame(transition_rows, columns=["timestamp", "from_aoi", "to_aoi", "task_id"])
    end_time = float(gaze["timestamp"].max()) if not gaze.empty else float(start_time + 10)
    events = pd.concat(
        [events, pd.DataFrame([{"timestamp": end_time, "type": "SESSION_END", "task_id": "none", "description": ""}])],
        ignore_index=True,
    )

    meta = {
        "session_id": session_id,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "total_gaze_records": int(len(gaze)),
        "total_fixations": int(len(fixations)),
        "total_transitions": int(len(transitions)),
        "total_blinks": int(len(blinks)),
        "total_tasks": int(len(tasks)),
    }
    return gaze, fixations, blinks, events, transitions, tasks, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 cognitive_data 格式的伪数据（用于跑通聚类流程）")
    p.add_argument("--out_root", type=str, default="cognitive_data_synth", help="输出根目录")
    p.add_argument("--n_sessions", type=int, default=30, help="生成多少个 session（建议 >= k）")
    p.add_argument("--seed", type=int, default=7, help="随机种子")
    p.add_argument("--n_tasks_min", type=int, default=1, help="每个 session 的任务数下限")
    p.add_argument("--n_tasks_max", type=int, default=3, help="每个 session 的任务数上限")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    base_epoch = 1769250000.0  # 随便给一个接近样例的 epoch 秒

    for i in range(int(args.n_sessions)):
        session_id = f"synth_{i+1:04d}"
        sdir = out_root / session_id
        n_tasks = int(rng.integers(int(args.n_tasks_min), int(args.n_tasks_max) + 1))
        start_time = base_epoch + i * 300.0 + float(rng.uniform(0, 2))
        pattern = int(i % 3)

        gaze, fix, blinks, events, trans, tasks, meta = _simulate_session(
            rng=rng,
            session_id=session_id,
            start_time=start_time,
            pattern=pattern,
            n_tasks=n_tasks,
        )

        _write_csv(gaze, sdir / "gaze_data.csv")
        _write_csv(fix, sdir / "fixations.csv")
        _write_csv(blinks, sdir / "blinks.csv")
        _write_csv(events, sdir / "events.csv")
        _write_csv(trans, sdir / "aoi_transitions.csv")
        _write_csv(tasks, sdir / "tasks.csv")
        _write_json(meta, sdir / "session_meta.json")

    print(f"[OK] synthetic cognitive_data written to: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

