#!/usr/bin/env python3
"""
Polymarket NVIDIA CEO Forecast Model 2 (Improved Pass):
Quantitative pricing forecast model computing quantile trajectories 4 hours ahead.
Estimates volatility using the population standard deviation of 1-minute price changes
over the last 120 minutes. Note: This model omits a drift term because Polymarket prices
are highly jumpy and mean-reverting; introducing any drift term empirically raised the
practice pinball loss. This distinguishes it from our first entry which used mean absolute
change over 60 minutes.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

WINDOW = 120
SIGMA_MULT = 1.0
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.003


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def market_sigma(prices):
    sub = prices[-(WINDOW + 1):]
    if len(sub) < 2:
        return SIGMA_FLOOR
    changes = [sub[i] - sub[i-1] for i in range(1, len(sub))]
    mean_c = sum(changes) / len(changes)
    pop_var = sum((c - mean_c) ** 2 for c in changes) / len(changes)
    sig = math.sqrt(pop_var) * SIGMA_MULT
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))


def forecast_market(prices, horizon):
    x = prices[-1]
    sig = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        # No drift term included: prices are jumpy and mean-reverting, and any drift
        # term was found to increase the practice pinball loss.
        spread = sig * math.sqrt(h)
        row = [price6(x + z * spread) for z in Z]
        for j in range(1, len(row)):
            if row[j] < row[j-1]:
                row[j] = row[j-1]
        rows.append(row)
    return rows


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    for file_path in in_files:
        with open(file_path, "r", encoding="utf-8") as f:
            case = json.load(f)

        if list(case["quantiles"]) != QUANTILES:
            raise ValueError("Quantiles mismatch")

        forecasts = []
        for m in case["markets"]:
            q_forecasts = forecast_market(m["prices"], case["horizon_minutes"])
            forecasts.append({
                "id": m["id"],
                "q": q_forecasts
            })

        out_file = output_dir / f"{file_path.stem}.out"
        output_data = {"forecasts": forecasts}
        with open(out_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(output_data) + "\n")


if __name__ == "__main__":
    main()
