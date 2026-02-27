from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .features import FeatureConfig


Unit = Literal["session", "task"]


@dataclass(frozen=True)
class CognitivePaths:
    gaze_data: str = "gaze_data.csv"
    fixations: str = "fixations.csv"
    blinks: str = "blinks.csv"
    events: str = "events.csv"
    aoi_transitions: str = "aoi_transitions.csv"
    tasks: str = "tasks.csv"
    session_meta: str = "session_meta.json"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gbk"))


def _entropy_from_counts(counts: np.ndarray) -> float:
    c = counts.astype("float64")
    s = c.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    p = c / s
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def _cat_stats(s: pd.Series) -> dict[str, float]:
    s = s.dropna().astype(str)
    if s.empty:
        return {"n_unique": 0.0, "entropy": float("nan"), "top1_prop": float("nan")}
    vc = s.value_counts(dropna=True)
    counts = vc.to_numpy()
    ent = _entropy_from_counts(counts)
    top1 = float(counts[0] / counts.sum()) if counts.size else float("nan")
    return {"n_unique": float(len(vc)), "entropy": float(ent), "top1_prop": top1}


def _bool01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype("int64")
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce") > 0).astype("int64")
    # 字符串 True/False
    v = s.astype(str).str.lower().map({"true": 1, "false": 0})
    return pd.to_numeric(v, errors="coerce").fillna(0).astype("int64")


def _time_range_seconds(ts: pd.Series) -> tuple[float, float, float]:
    t = pd.to_numeric(ts, errors="coerce").dropna().astype("float64")
    if t.empty:
        return float("nan"), float("nan"), 0.0
    t0 = float(t.min())
    t1 = float(t.max())
    dur = float(max(0.0, t1 - t0))
    return t0, t1, dur


def discover_sessions(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"找不到目录: {root}")

    # root 既可能是 cognitive_data/，也可能直接就是某个 session 文件夹
    if (root / "gaze_data.csv").exists() or (root / "session_meta.json").exists():
        return [root]

    sessions: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "gaze_data.csv").exists() and (p / "session_meta.json").exists():
            sessions.append(p)
    if not sessions:
        # 兜底：只要有 gaze_data.csv 也算
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "gaze_data.csv").exists():
                sessions.append(p)
    if not sessions:
        raise FileNotFoundError(f"在 {root} 下未发现 session 子目录（缺少 gaze_data.csv 等文件）")
    return sessions


def _extract_gaze_timeseries_features(
    gaze: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    cfg: FeatureConfig | None = None,
) -> dict[str, float]:
    if cfg is None:
        cfg = FeatureConfig()
    if gaze.empty or time_col not in gaze.columns:
        return {"gaze__n": 0.0, "gaze__duration": 0.0, "gaze__rate_hz": float("nan")}

    feats: dict[str, float] = {}
    feats["gaze__n"] = float(len(gaze))
    _, _, dur = _time_range_seconds(gaze[time_col])
    feats["gaze__duration"] = float(dur)
    feats["gaze__rate_hz"] = float(len(gaze) / dur) if dur and np.isfinite(dur) and dur > 0 else float("nan")

    # 数值列：按时序抽特征
    numeric_cols = [c for c in ["gaze_x", "gaze_y", "screen_x", "screen_y", "yaw", "pitch"] if c in gaze.columns]
    for c in numeric_cols:
        # 使用与 features.py 同风格的统计/差分/FFT/autocorr
        from .features import _series_features  # 延迟导入，避免循环

        for k, v in _series_features(gaze[c], gaze[time_col], cfg).items():
            feats[f"gaze__{c}__{k}"] = v

    # 布尔列：补充一些离散统计
    for bc in ["is_fixation", "is_luogu"]:
        if bc in gaze.columns:
            b = _bool01(gaze[bc]).to_numpy()
            feats[f"gaze__{bc}__mean"] = float(np.mean(b)) if b.size else float("nan")
            feats[f"gaze__{bc}__switches"] = float(np.sum(b[1:] != b[:-1])) if b.size >= 2 else 0.0

    # 类别列：AOI / task_id
    for cc in ["aoi_region", "aoi_name", "task_id"]:
        if cc in gaze.columns:
            st = _cat_stats(gaze[cc])
            feats[f"gaze__{cc}__n_unique"] = st["n_unique"]
            feats[f"gaze__{cc}__entropy"] = st["entropy"]
            feats[f"gaze__{cc}__top1_prop"] = st["top1_prop"]

    # fixation_id：用“唯一数/最大值”表征
    if "fixation_id" in gaze.columns:
        fid = pd.to_numeric(gaze["fixation_id"], errors="coerce").dropna()
        feats["gaze__fixation_id__n_unique"] = float(fid.nunique()) if not fid.empty else 0.0
        feats["gaze__fixation_id__max"] = float(fid.max()) if not fid.empty else float("nan")

    return feats


def _extract_fixation_features(fix: pd.DataFrame) -> dict[str, float]:
    if fix.empty:
        return {"fix__n": 0.0}
    feats: dict[str, float] = {"fix__n": float(len(fix))}
    if "duration" in fix.columns:
        d = pd.to_numeric(fix["duration"], errors="coerce").dropna().astype("float64")
        feats["fix__duration__sum"] = float(d.sum()) if not d.empty else 0.0
        feats["fix__duration__mean"] = float(d.mean()) if not d.empty else float("nan")
        feats["fix__duration__std"] = float(d.std(ddof=0)) if d.size >= 2 else float("nan")
        feats["fix__duration__median"] = float(d.median()) if not d.empty else float("nan")
    if "aoi_region" in fix.columns:
        st = _cat_stats(fix["aoi_region"])
        feats["fix__aoi_region__n_unique"] = st["n_unique"]
        feats["fix__aoi_region__entropy"] = st["entropy"]
        feats["fix__aoi_region__top1_prop"] = st["top1_prop"]
    return feats


def _extract_blink_features(blinks: pd.DataFrame, session_duration: float | None) -> dict[str, float]:
    if blinks.empty:
        return {"blink__n": 0.0}
    feats: dict[str, float] = {"blink__n": float(len(blinks))}
    if "ear" in blinks.columns:
        ear = pd.to_numeric(blinks["ear"], errors="coerce").dropna().astype("float64")
        feats["blink__ear__mean"] = float(ear.mean()) if not ear.empty else float("nan")
        feats["blink__ear__std"] = float(ear.std(ddof=0)) if ear.size >= 2 else float("nan")
        feats["blink__ear__min"] = float(ear.min()) if not ear.empty else float("nan")
    if session_duration is not None and np.isfinite(session_duration) and session_duration > 0:
        feats["blink__rate_per_s"] = float(len(blinks) / session_duration)
    return feats


def _extract_transition_features(tr: pd.DataFrame) -> dict[str, float]:
    if tr.empty:
        return {"trans__n": 0.0}
    feats: dict[str, float] = {"trans__n": float(len(tr))}
    for c in ["from_aoi", "to_aoi", "task_id"]:
        if c in tr.columns:
            st = _cat_stats(tr[c])
            feats[f"trans__{c}__n_unique"] = st["n_unique"]
            feats[f"trans__{c}__entropy"] = st["entropy"]
            feats[f"trans__{c}__top1_prop"] = st["top1_prop"]
    if "from_aoi" in tr.columns and "to_aoi" in tr.columns:
        same = (tr["from_aoi"].astype(str) == tr["to_aoi"].astype(str)).mean() if len(tr) else 0.0
        feats["trans__same_frac"] = float(same)
    return feats


def _extract_event_features(ev: pd.DataFrame) -> dict[str, float]:
    if ev.empty:
        return {"event__n": 0.0, "event__n_types": 0.0}
    feats: dict[str, float] = {"event__n": float(len(ev))}
    if "type" in ev.columns:
        vc = ev["type"].dropna().astype(str).value_counts()
        feats["event__n_types"] = float(len(vc))
        # 动态列：每种事件的计数（不同数据集可能类型不一样）
        for k, v in vc.items():
            feats[f"event__type__{k}__count"] = float(v)
    return feats


def _extract_task_features(tasks: pd.DataFrame) -> dict[str, float]:
    if tasks.empty:
        return {"task__n": 0.0}
    feats: dict[str, float] = {"task__n": float(len(tasks))}
    for c in ["duration", "difficulty", "subjective_difficulty", "subjective_effort"]:
        if c in tasks.columns:
            v = pd.to_numeric(tasks[c], errors="coerce").dropna().astype("float64")
            feats[f"task__{c}__mean"] = float(v.mean()) if not v.empty else float("nan")
            feats[f"task__{c}__std"] = float(v.std(ddof=0)) if v.size >= 2 else float("nan")
    if "result" in tasks.columns:
        vc = tasks["result"].dropna().astype(str).value_counts()
        feats["task__result__n_unique"] = float(len(vc))
        st = _cat_stats(tasks["result"])
        feats["task__result__entropy"] = st["entropy"]
    return feats


def extract_cognitive_features(
    root: str | Path,
    *,
    unit: Unit = "session",
    cfg: FeatureConfig | None = None,
    paths: CognitivePaths | None = None,
) -> pd.DataFrame:
    """
    适配 cognitive_data 格式：
    - root 下每个 session 子目录是一组数据（6 个 CSV + 1 个 JSON）
    - 输出：每个 session（或每个 task）一行的特征表
    """
    if cfg is None:
        cfg = FeatureConfig()
    if paths is None:
        paths = CognitivePaths()

    sessions = discover_sessions(root)
    rows: list[dict[str, float]] = []
    index: list[str] = []

    for sdir in sessions:
        meta = _safe_read_json(sdir / paths.session_meta) or {}
        gaze = _safe_read_csv(sdir / paths.gaze_data)
        fix = _safe_read_csv(sdir / paths.fixations)
        blinks = _safe_read_csv(sdir / paths.blinks)
        ev = _safe_read_csv(sdir / paths.events)
        tr = _safe_read_csv(sdir / paths.aoi_transitions)
        tasks = _safe_read_csv(sdir / paths.tasks)

        # session duration：优先 meta，否则从 gaze timestamp 推断
        session_dur: float | None = None
        if "start_time" in meta and "end_time" in meta:
            try:
                session_dur = float(meta["end_time"]) - float(meta["start_time"])
            except Exception:
                session_dur = None
        if session_dur is None and not gaze.empty and "timestamp" in gaze.columns:
            _, _, session_dur = _time_range_seconds(gaze["timestamp"])

        def build_base_features(
            gaze_sub: pd.DataFrame,
            fix_sub: pd.DataFrame,
            blinks_sub: pd.DataFrame,
            tr_sub: pd.DataFrame,
            ev_sub: pd.DataFrame,
            tasks_sub: pd.DataFrame,
            *,
            segment_duration: float | None,
            add_meta: bool,
        ) -> dict[str, float]:
            """
            根据传入的子数据表计算一段时间（一个 session 或一个 task）的特征。
            - gaze_sub / fix_sub / blinks_sub / tr_sub / ev_sub / tasks_sub 均应已按时间或 task_id 切好
            - segment_duration 用于 blink rate 等“单位时间”特征；对于 task 级别，尽量使用该 task 的持续时间
            - add_meta=True 时会把 session 级的 meta 统计（总帧数等）加入特征；仅适用于 unit=session
            """
            f: dict[str, float] = {}
            f.update(_extract_gaze_timeseries_features(gaze_sub, cfg=cfg))
            f.update(_extract_fixation_features(fix_sub))
            f.update(_extract_blink_features(blinks_sub, segment_duration))
            f.update(_extract_transition_features(tr_sub))
            f.update(_extract_event_features(ev_sub))
            f.update(_extract_task_features(tasks_sub))

            if add_meta:
                # meta 里的总量也加进来（如果有）
                for mk in [
                    "total_gaze_records",
                    "total_fixations",
                    "total_transitions",
                    "total_blinks",
                    "total_tasks",
                ]:
                    if mk in meta:
                        try:
                            f[f"meta__{mk}"] = float(meta[mk])
                        except Exception:
                            pass
                if session_dur is not None and np.isfinite(session_dur):
                    f["meta__session_duration"] = float(session_dur)
            return f

        session_id = sdir.name

        if unit == "session":
            rows.append(
                build_base_features(
                    gaze,
                    fix,
                    blinks,
                    tr,
                    ev,
                    tasks,
                    segment_duration=session_dur,
                    add_meta=True,
                )
            )
            index.append(session_id)
            continue

        # ------------ unit == "task"：按 task 级别切分 ------------

        # 1) 基础检查：若 gaze 没有 task_id，则退化为 session 级
        # （以保持向后兼容）
        if gaze.empty or "task_id" not in gaze.columns:
            rows.append(
                build_base_features(
                    gaze,
                    fix,
                    blinks,
                    tr,
                    ev,
                    tasks,
                    segment_duration=session_dur,
                    add_meta=False,
                )
            )
            index.append(f"{session_id}::task=__all__")
            continue

        task_ids = gaze["task_id"].dropna().astype(str).unique().tolist()
        # 过滤掉非真实任务的占位标记（例如浏览/休息段标记为 "none"）
        # 这些不应作为独立 task 参与聚类与监督训练
        task_ids = [tid for tid in task_ids if tid != "none"]
        if not task_ids:
            rows.append(
                build_base_features(
                    gaze,
                    fix,
                    blinks,
                    tr,
                    ev,
                    tasks,
                    segment_duration=session_dur,
                    add_meta=False,
                )
            )
            index.append(f"{session_id}::task=__all__")
            continue

        # 2) 从 tasks.csv 中构建每个 task 的时间窗口（start_time, end_time）
        task_time_map: dict[str, tuple[float | None, float | None, float | None]] = {}
        if not tasks.empty and "task_id" in tasks.columns:
            tdf = tasks.copy()
            tdf["_task_id_str"] = tdf["task_id"].astype(str)
            start_col = "start_time" if "start_time" in tdf.columns else None
            end_col = "end_time" if "end_time" in tdf.columns else None
            for _, row in tdf.iterrows():
                tid_str = str(row["_task_id_str"])
                t0: float | None = None
                t1: float | None = None
                dur: float | None = None
                if start_col is not None and end_col is not None:
                    try:
                        t0 = float(row[start_col])
                        t1 = float(row[end_col])
                        dur = max(0.0, t1 - t0)
                    except Exception:
                        t0, t1, dur = None, None, None
                # 如果没有 start/end 列，尝试使用 duration 字段
                if dur is None and "duration" in tdf.columns:
                    try:
                        dur = float(row["duration"])
                    except Exception:
                        dur = None
                task_time_map[tid_str] = (t0, t1, dur)

        # 3) 按 task_id / 时间窗口切分各个表
        for tid in sorted(task_ids):
            tid_str = str(tid)

            # --- gaze: 先按 task_id 切分，若为空再按时间窗口兜底 ---
            gaze_sub = gaze[gaze["task_id"].astype(str) == tid_str]
            t0: float | None = None
            t1: float | None = None
            tdur: float | None = None
            if tid_str in task_time_map:
                t0, t1, tdur = task_time_map[tid_str]

            if gaze_sub.empty and t0 is not None and t1 is not None and "timestamp" in gaze.columns:
                ts = pd.to_numeric(gaze["timestamp"], errors="coerce")
                gaze_sub = gaze[(ts >= t0) & (ts <= t1)]

            # 如果 tasks 表里有该 task 的独立行，则作为 task 级 features 的来源
            tasks_sub = pd.DataFrame()
            if not tasks.empty and "task_id" in tasks.columns:
                tasks_sub = tasks[tasks["task_id"].astype(str) == tid_str]

            # 若 tdur 仍为 None，则优先使用 tasks_sub["duration"]，再退化为 gaze_sub 的时间跨度
            if tdur is None:
                if not tasks_sub.empty and "duration" in tasks_sub.columns:
                    dv = pd.to_numeric(tasks_sub["duration"], errors="coerce").dropna().astype("float64")
                    tdur = float(dv.iloc[0]) if not dv.empty else None
                elif not gaze_sub.empty and "timestamp" in gaze_sub.columns:
                    _, _, gd = _time_range_seconds(gaze_sub["timestamp"])
                    tdur = gd

            # --- fixations: 优先按 task_id 切分，若没有 task_id 或为空则按时间区间（start_time/end_time 与 task 窗口有交集） ---
            fix_sub = pd.DataFrame()
            if not fix.empty:
                if "task_id" in fix.columns:
                    fix_sub = fix[fix["task_id"].astype(str) == tid_str]
                if fix_sub.empty and t0 is not None and t1 is not None and "start_time" in fix.columns:
                    st = pd.to_numeric(fix["start_time"], errors="coerce")
                    if "end_time" in fix.columns:
                        et = pd.to_numeric(fix["end_time"], errors="coerce")
                        mask = (st <= t1) & (et >= t0)
                    else:
                        mask = (st >= t0) & (st <= t1)
                    fix_sub = fix[mask]

            # --- blinks: 按 task_id 或 timestamp 区间 ---
            blinks_sub = pd.DataFrame()
            if not blinks.empty:
                if "task_id" in blinks.columns:
                    blinks_sub = blinks[blinks["task_id"].astype(str) == tid_str]
                if blinks_sub.empty and t0 is not None and t1 is not None and "timestamp" in blinks.columns:
                    bt = pd.to_numeric(blinks["timestamp"], errors="coerce")
                    blinks_sub = blinks[(bt >= t0) & (bt <= t1)]

            # --- AOI transitions: 按 task_id 或 timestamp 区间 ---
            tr_sub = pd.DataFrame()
            if not tr.empty:
                if "task_id" in tr.columns:
                    tr_sub = tr[tr["task_id"].astype(str) == tid_str]
                if tr_sub.empty and t0 is not None and t1 is not None and "timestamp" in tr.columns:
                    tt = pd.to_numeric(tr["timestamp"], errors="coerce")
                    tr_sub = tr[(tt >= t0) & (tt <= t1)]

            # --- events: 按 task_id 或 timestamp 区间 ---
            ev_sub = pd.DataFrame()
            if not ev.empty:
                if "task_id" in ev.columns:
                    ev_sub = ev[ev["task_id"].astype(str) == tid_str]
                if ev_sub.empty and t0 is not None and t1 is not None and "timestamp" in ev.columns:
                    et = pd.to_numeric(ev["timestamp"], errors="coerce")
                    ev_sub = ev[(et >= t0) & (et <= t1)]

            rows.append(
                build_base_features(
                    gaze_sub,
                    fix_sub,
                    blinks_sub,
                    tr_sub,
                    ev_sub,
                    tasks_sub,
                    segment_duration=tdur,
                    add_meta=False,  # task 级别不再注入 session 级 meta 统计，避免任务之间“串味”
                )
            )
            index.append(f"{session_id}::task={tid_str}")

    feat_df = pd.DataFrame(rows, index=pd.Index(index, name="sample_key"))
    return feat_df

