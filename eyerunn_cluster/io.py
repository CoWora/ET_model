from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ID_CANDIDATES: list[str] = [
    "sample_id",
    "sample",
    "trial_id",
    "trial",
    "segment_id",
    "subject_id",
    "subject",
    "sid",
    "id",
]

TIME_CANDIDATES: list[str] = [
    "timestamp",
    "time",
    "t",
    "ms",
    "millis",
    "frame",
    "frame_id",
    "index",
]


@dataclass(frozen=True)
class LoadInfo:
    csv_paths: list[str]
    json_path: str | None
    inferred_id_col: str | None
    inferred_time_col: str | None
    used_id_col: str
    used_time_col: str


def _first_existing(colnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _coerce_timestamp_to_numeric(s: pd.Series) -> pd.Series:
    # Already numeric?
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Try datetime parsing
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        # seconds since epoch
        return (dt.view("int64") / 1e9).astype("float64")

    # Fall back to numeric coercion from strings
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    # 常见编码兜底：utf-8 / gbk / utf-8-sig
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def load_multicsv_timeseries(
    data_dir: str | os.PathLike,
    *,
    csv_glob: str = "*.csv",
    json_path: str | os.PathLike | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
    prefix_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None, LoadInfo]:
    """
    读取 data_dir 下多个 CSV，并按 (sample_id, timestamp) 外连接对齐为同一张“长表”时序数据。

    返回:
    - merged_df: 列包含 sample_id, timestamp, 以及各类数值信号列
    - meta_json: JSON 内容（若存在）
    - info: 推断/使用的列信息
    """
    data_dir = Path(data_dir)
    csv_paths = sorted([p for p in data_dir.glob(csv_glob) if p.is_file()])
    if not csv_paths:
        raise FileNotFoundError(f"在 {data_dir} 下找不到匹配 {csv_glob} 的 CSV 文件")

    if json_path is None:
        json_candidates = sorted([p for p in data_dir.glob("*.json") if p.is_file()])
        json_path = json_candidates[0] if json_candidates else None

    meta_json: dict[str, Any] | None = None
    if json_path is not None:
        jp = Path(json_path)
        with jp.open("r", encoding="utf-8") as f:
            meta_json = json.load(f)

    dfs: list[pd.DataFrame] = []
    inferred_id_col: str | None = None
    inferred_time_col: str | None = None

    for p in csv_paths:
        df = _safe_read_csv(p)

        local_id = id_col or _first_existing(df.columns, ID_CANDIDATES)
        local_time = time_col or _first_existing(df.columns, TIME_CANDIDATES)

        if inferred_id_col is None and local_id is not None:
            inferred_id_col = local_id
        if inferred_time_col is None and local_time is not None:
            inferred_time_col = local_time

        # 标准化列名到统一 join key
        if local_id is None:
            df["sample_id"] = 0
            local_id = "sample_id"
        if local_time is None:
            df["timestamp"] = np.arange(len(df), dtype="float64")
            local_time = "timestamp"

        df = df.rename(columns={local_id: "sample_id", local_time: "timestamp"})

        # 类型清洗
        df["sample_id"] = pd.to_numeric(df["sample_id"], errors="ignore")
        df["timestamp"] = _coerce_timestamp_to_numeric(df["timestamp"])

        # 避免列名冲突：对信号列加前缀（文件名）
        if prefix_columns:
            stem = p.stem
            keep = {"sample_id", "timestamp"}
            rename_map = {}
            for c in df.columns:
                if c in keep:
                    continue
                # 只对重复/模糊列加前缀，尽量保留原名
                rename_map[c] = f"{stem}__{c}"
            df = df.rename(columns=rename_map)

        dfs.append(df)

    # 逐个外连接合并
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["sample_id", "timestamp"], how="outer")

    merged = merged.sort_values(["sample_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    info = LoadInfo(
        csv_paths=[str(p) for p in csv_paths],
        json_path=str(json_path) if json_path is not None else None,
        inferred_id_col=inferred_id_col,
        inferred_time_col=inferred_time_col,
        used_id_col="sample_id",
        used_time_col="timestamp",
    )
    return merged, meta_json, info

