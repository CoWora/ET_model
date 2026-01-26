from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    autocorr_lags: tuple[int, ...] = (1, 2, 5, 10)
    fft_bins: int = 64  # 过长序列会被下采样到该长度用于 FFT 特征


def _downsample_to_n(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return x[idx]


def _spectral_entropy(power: np.ndarray) -> float:
    p = power.astype("float64")
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum() / math.log(len(power)))


def _linear_slope(tv: np.ndarray, yv: np.ndarray) -> float:
    if yv.size < 2:
        return float("nan")
    t0 = tv.astype("float64")
    y0 = yv.astype("float64")
    # 防止数值不稳定：中心化
    t0 = t0 - np.nanmean(t0)
    y0 = y0 - np.nanmean(y0)
    denom = np.nansum(t0 * t0)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nansum(t0 * y0) / denom)


def _series_features(y: pd.Series, t: pd.Series, cfg: FeatureConfig) -> dict[str, float]:
    y = pd.to_numeric(y, errors="coerce")
    t = pd.to_numeric(t, errors="coerce")

    mask = y.notna() & t.notna()
    yv = y[mask].astype("float64").to_numpy()
    tv = t[mask].astype("float64").to_numpy()

    out: dict[str, float] = {}
    out["nan_frac"] = float(1.0 - (mask.mean() if len(mask) else 0.0))

    if yv.size == 0:
        # 全空时返回 NaN（除 nan_frac）
        for k in (
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
            "iqr",
            "skew",
            "kurt",
            "diff_mean",
            "diff_std",
            "absdiff_mean",
            "absdiff_std",
            "slope",
            "fft_dom_bin",
            "fft_spec_entropy",
        ):
            out[k] = float("nan")
        for lag in cfg.autocorr_lags:
            out[f"autocorr_lag{lag}"] = float("nan")
        return out

    s = pd.Series(yv)
    out["mean"] = float(s.mean())
    out["std"] = float(s.std(ddof=0))
    out["min"] = float(s.min())
    out["max"] = float(s.max())
    out["median"] = float(s.median())
    out["q25"] = float(s.quantile(0.25))
    out["q75"] = float(s.quantile(0.75))
    out["iqr"] = out["q75"] - out["q25"]
    out["skew"] = float(s.skew()) if yv.size >= 3 else float("nan")
    out["kurt"] = float(s.kurt()) if yv.size >= 4 else float("nan")

    if yv.size >= 2:
        d = np.diff(yv)
        out["diff_mean"] = float(np.nanmean(d))
        out["diff_std"] = float(np.nanstd(d))
        ad = np.abs(d)
        out["absdiff_mean"] = float(np.nanmean(ad))
        out["absdiff_std"] = float(np.nanstd(ad))
    else:
        out["diff_mean"] = float("nan")
        out["diff_std"] = float("nan")
        out["absdiff_mean"] = float("nan")
        out["absdiff_std"] = float("nan")

    out["slope"] = _linear_slope(tv, yv)

    # autocorr（使用 pandas 的定义）
    # 如果序列为常量（std=0），pandas/numpy 计算相关系数会产生 RuntimeWarning，这里直接返回 NaN。
    y_std = float(np.nanstd(yv))
    if not np.isfinite(y_std) or y_std <= 0:
        for lag in cfg.autocorr_lags:
            out[f"autocorr_lag{lag}"] = float("nan")
    else:
        ps = pd.Series(yv)
        for lag in cfg.autocorr_lags:
            if yv.size > lag + 1:
                try:
                    out[f"autocorr_lag{lag}"] = float(ps.autocorr(lag=lag))
                except Exception:
                    out[f"autocorr_lag{lag}"] = float("nan")
            else:
                out[f"autocorr_lag{lag}"] = float("nan")

    # FFT 特征：不依赖真实采样率，输出相对频率 bin
    y_fft = yv
    if y_fft.size >= 4:
        y_fft = _downsample_to_n(y_fft, cfg.fft_bins)
        y_fft = y_fft - np.nanmean(y_fft)
        spec = np.fft.rfft(y_fft)
        power = (spec.real ** 2 + spec.imag ** 2).astype("float64")
        if power.size > 1:
            # 去掉 DC 分量
            power_no_dc = power.copy()
            power_no_dc[0] = 0.0
            dom = int(np.argmax(power_no_dc))
            out["fft_dom_bin"] = float(dom / max(1, (power.size - 1)))
            out["fft_spec_entropy"] = _spectral_entropy(power_no_dc[1:] if power.size > 2 else power_no_dc)
        else:
            out["fft_dom_bin"] = float("nan")
            out["fft_spec_entropy"] = float("nan")
    else:
        out["fft_dom_bin"] = float("nan")
        out["fft_spec_entropy"] = float("nan")

    return out


def extract_features_per_sample(
    df: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    time_col: str = "timestamp",
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    输入：长表时序 df（包含 sample_id, timestamp 以及多列数值信号）
    输出：每个 sample_id 一行的特征表
    """
    if cfg is None:
        cfg = FeatureConfig()

    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"df 必须包含列 {id_col} 与 {time_col}")

    # 仅保留数值信号列
    signal_cols: list[str] = []
    for c in df.columns:
        if c in (id_col, time_col):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            signal_cols.append(c)

    if not signal_cols:
        raise ValueError("未找到任何数值信号列（除 sample_id/timestamp 之外）")

    rows: list[dict[str, float]] = []
    ids: list[object] = []
    for sid, g in df.groupby(id_col, sort=True):
        ids.append(sid)
        base = {"n_points": float(len(g))}
        t = g[time_col]
        for c in signal_cols:
            feats = _series_features(g[c], t, cfg)
            for k, v in feats.items():
                base[f"{c}__{k}"] = v
        rows.append(base)

    feat_df = pd.DataFrame(rows, index=pd.Index(ids, name=id_col))
    return feat_df

