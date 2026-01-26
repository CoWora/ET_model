from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    out = Path("data")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    n_samples = 100

    frames: list[pd.DataFrame] = []
    meta = {"n_samples": n_samples, "note": "synthetic eye-tracking demo", "samples": {}}

    for sid in range(n_samples):
        n = int(rng.integers(220, 420))
        t = np.cumsum(rng.uniform(0.01, 0.02, size=n))  # ~50-100Hz irregular-ish

        # 3 个“潜在簇”模式
        cluster = int(sid % 3)
        base_freq = [1.0, 2.5, 0.5][cluster]
        noise = [0.15, 0.25, 0.12][cluster]

        gaze_x = np.sin(2 * np.pi * base_freq * t) + rng.normal(0, noise, size=n)
        gaze_y = np.cos(2 * np.pi * base_freq * t) + rng.normal(0, noise, size=n)
        pupil = 3.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t + cluster) + rng.normal(0, 0.05, size=n)

        # blink：稀疏脉冲
        blink = (rng.uniform(size=n) < (0.01 + 0.005 * cluster)).astype(int)

        # velocity：由 gaze 差分近似
        vx = np.concatenate([[0.0], np.diff(gaze_x)])
        vy = np.concatenate([[0.0], np.diff(gaze_y)])
        velocity = np.sqrt(vx * vx + vy * vy)

        # fixation：简化为分段 id
        fixation_id = np.repeat(np.arange((n + 24) // 25), 25)[:n]

        df = pd.DataFrame(
            {
                "sample_id": sid,
                "timestamp": t,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "pupil": pupil,
                "blink": blink,
                "velocity": velocity,
                "fixation_id": fixation_id,
            }
        )
        frames.append(df)
        meta["samples"][str(sid)] = {"synthetic_cluster": cluster, "n_points": n}

    all_df = pd.concat(frames, ignore_index=True)

    # 拆成 6 个 CSV（模拟“多源 CSV”）
    all_df[["sample_id", "timestamp", "gaze_x", "gaze_y"]].to_csv(out / "01_gaze.csv", index=False)
    all_df[["sample_id", "timestamp", "pupil"]].to_csv(out / "02_pupil.csv", index=False)
    all_df[["sample_id", "timestamp", "blink"]].to_csv(out / "03_blink.csv", index=False)
    all_df[["sample_id", "timestamp", "velocity"]].to_csv(out / "04_velocity.csv", index=False)
    all_df[["sample_id", "timestamp", "fixation_id"]].to_csv(out / "05_fixation.csv", index=False)
    # 再造一个“事件”表
    event = all_df[["sample_id", "timestamp"]].copy()
    event["event_flag"] = (rng.uniform(size=len(event)) < 0.02).astype(int)
    event.to_csv(out / "06_event.csv", index=False)

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] synthetic data written to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

