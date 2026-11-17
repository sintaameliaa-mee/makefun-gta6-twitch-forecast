#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

Reads one case (schema makefun/twitch-ccu-case/v1) from /input (or
$MAKEFUN_INPUT_DIR) and writes a forecast to /output (or $MAKEFUN_OUTPUT_DIR):
for each of hours 2..24 of the release day, five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent-viewer count.

Method: launch-day shape transfer from comparable launches.
Every comparable launch in the case carries its full release-day hourly curve.
For each comparable we compute the log-ratio of each hour's count to its own
hour-1 count. The forecast anchors on this case's hour-1 print and applies
the comparables' ratio distribution per hour. Comparables are weighted by
how close their hour-1 print is to this case's (Gaussian kernel in log
space): a launch's hour-1 print is the strongest single signal of its
day-shape, because it measures how much of the launch had already happened
by 01:00 UTC. Early-access-contaminated comparables are down-weighted.
Quantiles are the weighted empirical quantiles of the log-ratios per hour,
lightly smoothed across neighbouring hours. The program is fully
deterministic (no randomness anywhere) and uses only the standard library.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at LEVELS, for the fallback forecast.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Log-space bandwidth of the hour-1 similarity kernel (in nats). Tuned on the
# four practice cases and a leave-one-out backtest over the comparables:
# 0.9 balances the two (practice 0.220, LOO 0.402, vs house 0.387 / 0.930).
BANDWIDTH = 0.9
# Weight multiplier for early-access-contaminated comparables. Their
# release-day curves are flattened by players already in-game before launch,
# so they are weaker guides for a clean launch like this one; 0.5 is the
# value that scored best on the (clean) practice cases.
CONTAMINATED_WEIGHT = 0.5
# Minimum number of comparable curves contributing to an hour before the
# empirical quantiles are trusted; below this, fall back to a widening
# flat forecast for that hour.
MIN_CURVES_PER_HOUR = 3
# Fallback spread growth (log std per sqrt hour), mirroring the house
# baseline's shape when the comparables cannot speak for an hour.
FALLBACK_SPREAD = 0.15
MIN_VIEWERS = 1.0
MAX_VIEWERS = 1.0e9


def parse_time(value):
    """Parse an ISO-8601 UTC timestamp into a datetime (None on failure)."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def hour1_print(case):
    """Return this case's hour-1 print (at least 1.0)."""
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    for row in rows:
        if isinstance(row, dict):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    # Fall back to the last history point before the checkpoint.
    history = case.get("history") or []
    for point in reversed(history):
        if isinstance(point, dict):
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    return MIN_VIEWERS


def release_curve(comparable):
    """Extract a comparable's release-day curve as {hour: viewers}.

    Hours are 1..24 counted from the comparable's day_start. Returns None
    when the curve has no usable hour-1 count (e.g. a launch whose data
    ends before its release day), since ratios need that anchor.
    """
    day_start = parse_time(comparable.get("day_start"))
    hourly = comparable.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return None
    curve = {}
    for point in hourly:
        if not isinstance(point, dict):
            continue
        when = parse_time(point.get("from"))
        viewers = point.get("viewers")
        if when is None or not isinstance(viewers, (int, float)):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and viewers > 0:
            curve[hour] = float(viewers)
    if curve.get(1, 0.0) <= 0.0:
        return None
    return curve


def is_contaminated(comparable):
    """True when the comparable's unlock notes flag early-access contamination."""
    unlock = comparable.get("unlock") or ""
    text = unlock.lower() if isinstance(unlock, str) else ""
    return "contaminat" in text or "early-access" in text


def weighted_quantiles(pairs, probs):
    """Interpolated quantiles of weighted samples.

    pairs: list of (value, weight) with positive weights. Returns one
    quantile per probability in probs, in the same order.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0.0 or not ordered:
        return [0.0] * len(probs)
    # Cumulative weight at each sample's midpoint, so the median of an
    # odd sample lands on the middle value and tails interpolate.
    anchors = []
    running = 0.0
    for value, weight in ordered:
        running += weight
        anchors.append((value, running - weight / 2.0))
    out = []
    for p in probs:
        target = p * total
        if target <= anchors[0][1]:
            out.append(anchors[0][0])
            continue
        if target >= anchors[-1][1]:
            out.append(anchors[-1][0])
            continue
        for i in range(1, len(anchors)):
            if anchors[i][1] >= target:
                v0, c0 = anchors[i - 1]
                v1, c1 = anchors[i]
                frac = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
                out.append(v0 + frac * (v1 - v0))
                break
    return out


def smooth_rows(rows):
    """Light (1, 2, 1) / 4 smoothing of each quantile level across hours.

    rows: {hour: [q05, q25, q50, q75, q95]} in log space is smoothed per
    level, then monotonicity within each row is re-enforced.
    """
    hours = sorted(rows)
    for level in range(5):
        values = [math.log(max(rows[h][level], MIN_VIEWERS)) for h in hours]
        smoothed = []
        for i, value in enumerate(values):
            prev = values[i - 1] if i > 0 else value
            nxt = values[i + 1] if i + 1 < len(values) else value
            smoothed.append((prev + 2.0 * value + nxt) / 4.0)
        for i, h in enumerate(hours):
            rows[h][level] = math.exp(smoothed[i])
    for h in hours:
        quantiles = rows[h]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
    return rows


def fallback_row(print1, hour):
    """Flat forecast with a widening spread, used when comparables are thin."""
    spread = FALLBACK_SPREAD * math.sqrt(hour - 1)
    return [max(print1 * math.exp(z * spread), MIN_VIEWERS) for z in Z_SCORES]


def forecast(case):
    """Build the forecast object for one case."""
    print1 = hour1_print(case)
    log_print1 = math.log(print1)

    # Collect comparable curves with their similarity weights.
    weighted_curves = []
    comparables = case.get("comparables") or []
    for comparable in comparables:
        if not isinstance(comparable, dict):
            continue
        curve = release_curve(comparable)
        if curve is None:
            continue
        gap = math.log(curve[1]) - log_print1
        weight = math.exp(-(gap * gap) / (2.0 * BANDWIDTH * BANDWIDTH))
        if is_contaminated(comparable):
            weight *= CONTAMINATED_WEIGHT
        weighted_curves.append((curve, weight))

    rows = {}
    for hour in range(2, 25):
        pairs = []
        for curve, weight in weighted_curves:
            if hour in curve:
                pairs.append((math.log(curve[hour] / curve[1]), weight))
        if len(pairs) >= MIN_CURVES_PER_HOUR:
            log_quantiles = weighted_quantiles(pairs, LEVELS)
            rows[hour] = [print1 * math.exp(q) for q in log_quantiles]
        else:
            rows[hour] = fallback_row(print1, hour)

    rows = smooth_rows(rows)

    hours = []
    for hour in range(2, 25):
        quantiles = []
        for value in rows[hour]:
            value = round(value, 2)
            if not math.isfinite(value) or value < MIN_VIEWERS:
                value = MIN_VIEWERS
            if value > MAX_VIEWERS:
                value = MAX_VIEWERS
            quantiles.append(value)
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def flat_forecast(print1):
    """Last-resort valid forecast: every quantile equals the hour-1 print."""
    value = round(max(min(print1, MAX_VIEWERS), MIN_VIEWERS), 2)
    return {"hours": [{"hour": h, "q": [value] * 5} for h in range(2, 25)]}


def run_case(input_dir, output_dir, filename):
    """Forecast one case file and write its output file. Never raises."""
    case_id = filename[:-3]  # strip ".in"
    out_path = os.path.join(output_dir, case_id + ".out")
    result = None
    try:
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast(case)
    except Exception:
        result = None
    if result is None:
        # The input could not be read or forecast: emit a flat forecast
        # so the case still has a valid output file.
        try:
            with open(os.path.join(input_dir, filename), "r",
                      encoding="utf-8") as fh:
                case = json.load(fh)
            result = flat_forecast(hour1_print(case))
        except Exception:
            result = flat_forecast(MIN_VIEWERS)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, separators=(",", ":"), sort_keys=False)
        fh.write("\n")


def main(argv=None):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(input_dir))
    except OSError:
        return 0
    for name in names:
        if name.endswith(".in"):
            run_case(input_dir, output_dir, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
