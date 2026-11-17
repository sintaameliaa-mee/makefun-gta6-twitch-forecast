#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

Reads one case (schema makefun/twitch-ccu-case/v1) from /input (or
$MAKEFUN_INPUT_DIR) and writes a forecast to /output (or $MAKEFUN_OUTPUT_DIR):
for each of hours 2..24 of the release day, five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent-viewer count.

Method: ridge regression over comparable launches, per hour.
For each hour, log(viewers) is regressed on two predictors across the
comparable launches: the log of the launch's hour-1 print, and the log
of its pre-launch 24-hour peak (which catches hour-1 prints that sit
in a timezone trough, where the launch was already bigger the day
before). The hour-1 slope is shrunk toward 1 and the peak slope toward
0, so with no evidence the model is the plain ratio transfer; the
quantiles come from a Gaussian fit to the weighted residuals. The
comparables are weighted by hour-1 similarity (Gaussian kernel, with
the bandwidth widened when the target is extreme so at least eight
comparables stay in reach) and by how well they match the target
launch: same early-access contamination status, same unlock pattern
(rolling midnight-local vs a single global unlock). Quantiles are
lightly smoothed across neighbouring hours. The program is fully
deterministic (no randomness anywhere) and uses only the standard
library.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at LEVELS, for the Gaussian fits and the
# fallback forecast.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Log-space bandwidth of the hour-1 similarity kernel (in nats).
BANDWIDTH = 0.9
# The kernel widens, when needed, so at least this many comparables
# sit within one bandwidth of the target's hour-1 print: an extreme
# target (a launch bigger than anything before it) still gets a
# usable neighbourhood instead of one or two neighbours.
ADAPTIVE_K = 8
# Weight multiplier for a comparable whose early-access contamination
# status differs from the target's: a contaminated launch is a good
# analogue for a contaminated target and a poor one for a clean one.
CONTAM_MISMATCH = 0.3
# Weight multiplier for a comparable whose unlock pattern differs
# from the target's (only when both patterns are known).
UNLOCK_MISMATCH = 0.5
# Ridge strength (in effective-sample units) shrinking the hour-1
# slope toward 1 (the ratio model).
REGRESSION_PRIOR = 5.0
# Ridge strength shrinking the pre-launch-peak slope toward 0 (no
# peak effect). The peak helps a little, so it is shrunk hard.
PEAK_PRIOR = 30.0
# Pre-launch window in hours before day_start, for the peak.
PRELAUNCH_HOURS = 24
# Floor on the Gaussian log-spread, to avoid degenerate narrow bands.
GAUSSIAN_MIN_SD = 0.05
# Minimum number of comparable curves contributing to an hour before
# the regression is trusted; below this, fall back to a widening
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
    ends before its release day), since the regression needs that anchor.
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


def unlock_features(text):
    """Parse an unlock/notes string into (contaminated, pattern).

    The Tier field wins where present (comparables carry it); prose
    parsing with a negation guard covers the case notes, which don't.
    pattern is "rolling", "simultaneous" or "unknown".
    """
    t = (text or "").lower() if isinstance(text, str) else ""
    if "tier: early-access-contaminated" in t:
        contaminated = True
    elif "tier:" in t:
        contaminated = False
    else:
        contaminated = (("early access" in t or "early-access" in t)
                        and "no early access" not in t)
    if "midnight local" in t or "rolling" in t:
        pattern = "rolling"
    elif ("simultaneous" in t or "single global" in t
            or "single time" in t):
        pattern = "simultaneous"
    else:
        pattern = "unknown"
    return contaminated, pattern


def hourly_points(case):
    """The target's pre-checkpoint hourly points as {datetime: viewers}.

    Prefers history_hourly (whole-hour means); falls back to the raw
    history aggregated by hour (the final case's minute reads).
    """
    points = {}
    hourly = case.get("history_hourly")
    if isinstance(hourly, list) and hourly:
        for row in hourly:
            if not isinstance(row, dict):
                continue
            when = parse_time(row.get("from"))
            viewers = row.get("viewers")
            if (when is not None and isinstance(viewers, (int, float))
                    and viewers > 0):
                points[when] = float(viewers)
        if points:
            return points
    history = case.get("history") or []
    buckets = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        when = parse_time(row.get("at"))
        viewers = row.get("viewers")
        if (when is None or not isinstance(viewers, (int, float))
                or viewers <= 0):
            continue
        hour = when.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append(float(viewers))
    for hour, values in buckets.items():
        points[hour] = sum(values) / len(values)
    return points


def pre_peak(points, day_start, hour1):
    """Log of the highest hourly viewers in the 24 h before day_start.

    points: {datetime: viewers}. Falls back to log(hour1) when there
    is no pre-launch data (a neutral 'no pre-surge detected').
    """
    peak = 0.0
    for when, viewers in points.items():
        offset = int((when - day_start).total_seconds() // 3600)
        if -PRELAUNCH_HOURS <= offset <= -1 and viewers > peak:
            peak = viewers
    if peak <= 0.0:
        return math.log(max(hour1, MIN_VIEWERS))
    return math.log(peak)


def gaussian_quantiles(pairs, probs):
    """Quantiles of a weighted Gaussian fit to the samples."""
    total = sum(w for _, w in pairs)
    if total <= 0.0 or not pairs:
        return [0.0] * len(probs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    sd = max(math.sqrt(var), GAUSSIAN_MIN_SD)
    return [mean + z * sd for z in Z_SCORES]


def regression_quantiles(quads, log_print1, log_peak):
    """Two-predictor ridge regression quantiles of log(viewers).

    quads: (log hour-1 print, log pre-launch peak, log hour-h
    viewers, weight) per comparable. Fits y ~ a + b1*x1 + b2*x2 with
    b1 shrunk toward 1 and b2 toward 0 (the plain ratio transfer),
    then adds a Gaussian fit of the residuals to the fitted median.
    Returns absolute log-viewer quantiles.
    """
    total = sum(w for _, _, _, w in quads)
    if total <= 0.0 or len(quads) < MIN_CURVES_PER_HOUR:
        return [log_print1] * len(LEVELS)
    x1bar = sum(x1 * w for x1, _, _, w in quads) / total
    x2bar = sum(x2 * w for _, x2, _, w in quads) / total
    ybar = sum(y * w for _, _, y, w in quads) / total
    s11 = sum(w * (x1 - x1bar) ** 2 for x1, _, _, w in quads) / total
    s12 = sum(w * (x1 - x1bar) * (x2 - x2bar)
              for x1, x2, _, w in quads) / total
    s22 = sum(w * (x2 - x2bar) ** 2 for _, x2, _, w in quads) / total
    s1y = sum(w * (x1 - x1bar) * (y - ybar)
              for x1, _, y, w in quads) / total
    s2y = sum(w * (x2 - x2bar) * (y - ybar)
              for _, x2, y, w in quads) / total
    n_eff = total * total / sum(w * w for _, _, _, w in quads)
    k1 = REGRESSION_PRIOR / max(n_eff, 1.0)
    k2 = PEAK_PRIOR / max(n_eff, 1.0)
    a11, a12, a22 = s11 + k1, s12, s22 + k2
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        b1, b2 = 1.0, 0.0
    else:
        b1 = ((s1y + k1) * a22 - s2y * a12) / det
        b2 = (a11 * s2y - a12 * (s1y + k1)) / det
    intercept = ybar - b1 * x1bar - b2 * x2bar
    median = intercept + b1 * log_print1 + b2 * log_peak
    # The fitted line passes through the weighted mean point, so the
    # weighted mean residual is zero by construction; the Gaussian
    # fit of the residuals is already centred on the median.
    residuals = [(y - (intercept + b1 * x1 + b2 * x2), w)
                 for x1, x2, y, w in quads]
    spread = gaussian_quantiles(residuals, LEVELS)
    return [median + s for s in spread]


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
    day_start = parse_time(case.get("day_start"))

    # Target-side features: unlock pattern/contamination from the
    # notes, pre-launch peak from the hourly history.
    t_contam, t_pattern = unlock_features(case.get("notes") or "")
    t_log_peak = log_print1
    if day_start is not None:
        t_log_peak = pre_peak(hourly_points(case), day_start, print1)

    # Collect comparable curves with their similarity factors. The
    # hour-1 kernel is applied after the loop so the bandwidth can
    # adapt to how far the nearest comparables are.
    collected = []
    comparables = case.get("comparables") or []
    for comparable in comparables:
        if not isinstance(comparable, dict):
            continue
        curve = release_curve(comparable)
        if curve is None:
            continue
        gap = math.log(curve[1]) - log_print1
        factor = 1.0
        c_contam, c_pattern = unlock_features(
            comparable.get("unlock") or "")
        if c_contam != t_contam:
            factor *= CONTAM_MISMATCH
        if c_pattern != "unknown" and t_pattern != "unknown" \
                and c_pattern != t_pattern:
            factor *= UNLOCK_MISMATCH
        c_log_peak = math.log(curve[1])
        c_day = parse_time(comparable.get("day_start"))
        if c_day is not None:
            c_points = {}
            hourly = comparable.get("hourly") or []
            for point in hourly:
                if not isinstance(point, dict):
                    continue
                when = parse_time(point.get("from"))
                viewers = point.get("viewers")
                if (when is not None and isinstance(viewers, (int, float))
                        and viewers > 0):
                    c_points[when] = float(viewers)
            if c_points:
                c_log_peak = pre_peak(c_points, c_day, curve[1])
        collected.append((curve, gap, factor, c_log_peak))

    bandwidth = BANDWIDTH
    if len(collected) >= ADAPTIVE_K:
        gaps = sorted(abs(gap) for _, gap, _, _ in collected)
        bandwidth = max(BANDWIDTH, gaps[ADAPTIVE_K - 1])
    weighted_curves = []
    for curve, gap, factor, c_log_peak in collected:
        weight = (math.exp(-(gap * gap) / (2.0 * bandwidth * bandwidth))
                  * factor)
        weighted_curves.append((curve, weight, c_log_peak))

    rows = {}
    for hour in range(2, 25):
        quads = [(math.log(curve[1]), c_log_peak, math.log(curve[hour]),
                  weight)
                 for curve, weight, c_log_peak in weighted_curves
                 if hour in curve]
        if len(quads) >= MIN_CURVES_PER_HOUR:
            log_quantiles = regression_quantiles(
                quads, log_print1, t_log_peak)
            rows[hour] = [math.exp(q) for q in log_quantiles]
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
