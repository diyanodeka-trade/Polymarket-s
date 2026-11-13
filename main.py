#!/usr/bin/env python3
"""
Polymarket NVIDIA CEO Forecast Model v3 (Improved Pass)
------------------------------------------------------
Generates 5 quantiles per minute over a 4-hour horizon (240 minutes) for NVIDIA CEO prediction markets.

Model design & choices:
1. Two-window adaptive volatility: the two-window blend tracks fresh volatility and the recent regime.
2. Asymmetric spread: the upside is scaled by 0.8 because future moves are skewed down.
3. Jump-reversion median path: the median reverts half of a one-minute jump because prices that spike usually pull partway back — and there is no plain drift term, because every drift / mean-reversion anchor tested raised the practice pinball loss.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

MEANABS_TO_SIGMA = 1.2533
SIGMA_MULT = 1.4
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005
W_FAST = 30
W_SLOW = 120
W_FAST_WEIGHT = 0.5
UP_SCALE = 0.8
DOWN_SCALE = 1.0
JUMP_THRESH = 0.03
JUMP_REVERT = 0.5
JUMP_HALF_LIFE = 30.0


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def mean_abs_change(prices, window):
    sub = prices[-(window + 1):]
    if len(sub) < 2:
        return 0.0
    diffs = [abs(sub[i] - sub[i - 1]) for i in range(1, len(sub))]
    return sum(diffs) / len(diffs)


def market_sigma(prices):
    # Two-window blend: tracks fresh volatility and the recent regime
    fast = mean_abs_change(prices, W_FAST)
    slow = mean_abs_change(prices, W_SLOW)
    blended = W_FAST_WEIGHT * fast + (1.0 - W_FAST_WEIGHT) * slow
    sig = blended * MEANABS_TO_SIGMA * SIGMA_MULT
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sig))


def median_path(prices, horizon):
    x = prices[-1]
    d = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    # Jump reversion: median reverts half of a one-minute jump because prices that spike usually pull partway back
    if abs(d) > JUMP_THRESH:
        return [x - JUMP_REVERT * d * (1.0 - math.exp(-h / JUMP_HALF_LIFE)) for h in range(1, horizon + 1)]
    else:
        return [x] * horizon


def forecast_market(prices, horizon):
    sig = market_sigma(prices)
    med = median_path(prices, horizon)
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        m = med[h - 1]
        # Asymmetric spread: upside is scaled by 0.8 because future moves are skewed down
        row = [price6(m + z * spread * (UP_SCALE if z > 0 else DOWN_SCALE)) for z in Z]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(input_dir.glob("*.in")):
        with open(p, "r", encoding="utf-8") as f:
            case = json.load(f)

        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"Invalid quantiles in {p.name}")

        horizon = case["horizon_minutes"]
        forecasts = []
        for m in case["markets"]:
            q_rows = forecast_market(m["prices"], horizon)
            forecasts.append({"id": m["id"], "q": q_rows})

        out_data = {"forecasts": forecasts}
        case_id = case.get("case", p.stem)
        out_path = output_dir / f"{case_id}.out"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out_data) + "\n")


if __name__ == "__main__":
    main()
