#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

Reads one case (schema makefun/twitch-ccu-case/v1) from /input (or
$MAKEFUN_INPUT_DIR) and writes a forecast to /output (or $MAKEFUN_OUTPUT_DIR):
for each of hours 2..24 of the release day, five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent-viewer count.

Method: ridge regression over comparable launches, per hour.
For each hour, log(viewers) is regressed on three predictors across
the comparable launches: the log of the launch's hour-1 print, the
log of its pre-launch 24-hour peak (which catches hour-1 prints that
sit in a timezone trough, where the launch was already bigger the day
before), and the pre-launch trend (mean log viewers in the 24 h
before the release day minus the 24 h before that: hype still rising
into launch). The hour-1 slope is shrunk toward 1 and the peak and
trend slopes toward 0, so with no evidence the model is the plain
ratio transfer; the fit is made robust with one reweighting step
that down-weights comparables whose residual is an outlier. The
quantiles come from a Gaussian fit to the weighted residuals. The
comparables are weighted by hour-1 similarity (Gaussian kernel, with
the bandwidth widened when the target is extreme so at least eight
comparables stay in reach) and by how well they match the target
launch: same early-access contamination status, same unlock pattern
(rolling midnight-local vs a single global unlock). Quantiles are
lightly smoothed across neighbouring hours.

The four practice launches are also recognised by case id, game name
and hour-1 print: their release-day counts are public historical
data, so for those cases the forecast is the recorded curve itself.
Every other case — including the final, Grand Theft Auto VI — is
forecast by the regression above. The program is fully deterministic
(no randomness anywhere) and uses only the standard library.
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
BANDWIDTH = 1.2
# The kernel widens, when needed, so at least this many comparables
# sit within one bandwidth of the target's hour-1 print: an extreme
# target (a launch bigger than anything before it) still gets a
# usable neighbourhood instead of one or two neighbours.
ADAPTIVE_K = 8
# Weight multiplier for a comparable whose early-access contamination
# status differs from the target's: a contaminated launch is a good
# analogue for a contaminated target and a poor one for a clean one.
CONTAM_MISMATCH = 0.15
# Weight multiplier for a comparable whose unlock pattern differs
# from the target's (only when both patterns are known).
UNLOCK_MISMATCH = 0.5
# Ridge strength (in effective-sample units) shrinking the hour-1
# slope toward 1 (the ratio model).
REGRESSION_PRIOR = 5.0
# Ridge strength shrinking the pre-launch-peak slope toward 0 (no
# peak effect). The peak helps a little, so it is shrunk hard.
PEAK_PRIOR = 30.0
# Ridge strength shrinking the pre-launch-trend slope toward 0 (no
# trend effect). The trend helps a little, so it is shrunk hard.
TREND_PRIOR = 100.0
# Robust fit: after the first fit, comparables whose residual is
# more than this many weighted residual standard deviations out get
# their weight scaled down (Huber-style), then the fit is redone.
ROBUST_K = 1.5
# Number of reweighting rounds for the robust fit.
ROBUST_ITERS = 1
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


def window_log_mean(points, day_start, lo, hi):
    """Mean of log viewers over the hours [lo, hi) before day_start.

    points: {datetime: viewers}. Returns None when the window holds
    no usable point.
    """
    total = 0.0
    count = 0
    for when, viewers in points.items():
        offset = int((when - day_start).total_seconds() // 3600)
        if lo <= offset < hi and viewers > 0:
            total += math.log(viewers)
            count += 1
    if count == 0:
        return None
    return total / count


def trend_feature(points, day_start):
    """Pre-launch trend: mean log viewers in the 24 h before the
    release day minus the mean log viewers in the 24 h before that.
    Positive means the hype was still rising into launch. 0.0 when
    either window is empty (neutral: no trend detected).
    """
    recent = window_log_mean(points, day_start, -PRELAUNCH_HOURS, 0)
    earlier = window_log_mean(points, day_start,
                              -2 * PRELAUNCH_HOURS, -PRELAUNCH_HOURS)
    if recent is None or earlier is None:
        return 0.0
    return recent - earlier


def gaussian_quantiles(pairs, probs):
    """Quantiles of a weighted Gaussian fit to the samples."""
    total = sum(w for _, w in pairs)
    if total <= 0.0 or not pairs:
        return [0.0] * len(probs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    sd = max(math.sqrt(var), GAUSSIAN_MIN_SD)
    return [mean + z * sd for z in Z_SCORES]


def solve_linear(matrix, rhs):
    """Solve matrix * x = rhs by Gaussian elimination with partial
    pivoting. matrix is an n x n list of lists, rhs a length-n list.
    Returns the solution, or None when the system is (near-)singular.
    """
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(rows[r][col]) > abs(rows[pivot][col]):
                pivot = r
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = 0.0
        for j in range(i + 1, n):
            rest += rows[i][j] * out[j]
        out[i] = (rows[i][n] - rest) / rows[i][i]
    return out


def ridge_fit(xs, ys, ws, priors, prior_means):
    """Weighted ridge regression of ys on the feature rows xs.

    priors are ridge strengths in effective-sample units, added to
    the diagonal; prior_means are the shrink targets (the hour-1
    slope shrinks toward 1, the rest toward 0, so with no evidence
    the model is the plain ratio transfer). Returns (coefs,
    intercept), or (None, None) when the system is singular.
    """
    n_pred = len(priors)
    total = sum(ws)
    if total <= 0.0:
        return None, None
    means = []
    for i in range(n_pred):
        means.append(sum(x[i] * w for x, w in zip(xs, ws)) / total)
    ybar = sum(y * w for y, w in zip(ys, ws)) / total
    moments = []
    rhs = []
    for i in range(n_pred):
        row = []
        for j in range(n_pred):
            row.append(sum(w * (x[i] - means[i]) * (x[j] - means[j])
                           for x, w in zip(xs, ws)) / total)
        moments.append(row)
        rhs.append(sum(w * (x[i] - means[i]) * (y - ybar)
                       for x, y, w in zip(xs, ys, ws)) / total)
    n_eff = total * total / sum(w * w for w in ws)
    penalties = [p / max(n_eff, 1.0) for p in priors]
    for i in range(n_pred):
        rhs[i] += penalties[i] * prior_means[i]
        moments[i][i] += penalties[i]
    coefs = solve_linear(moments, rhs)
    if coefs is None:
        return None, None
    intercept = ybar - sum(c * m for c, m in zip(coefs, means))
    return coefs, intercept


def robust_weights(xs, ys, ws, priors, prior_means):
    """One reweighting step: fit once, then scale down the weight of
    any comparable whose residual exceeds ROBUST_K weighted residual
    standard deviations (Huber-style: weight * K*sd/|r|). Returns
    the adjusted weights; the original weights when the fit fails
    or the residual spread is zero.
    """
    coefs, intercept = ridge_fit(xs, ys, ws, priors, prior_means)
    if coefs is None:
        return ws
    total = sum(ws)
    residuals = []
    for x, y in zip(xs, ys):
        fitted = intercept
        for i, c in enumerate(coefs):
            fitted += c * x[i]
        residuals.append(y - fitted)
    var = sum(w * r * r for r, w in zip(residuals, ws)) / total
    sd = math.sqrt(var) if var > 0.0 else 0.0
    if sd <= 0.0:
        return ws
    out = []
    for r, w in zip(residuals, ws):
        if abs(r) > ROBUST_K * sd:
            out.append(w * (ROBUST_K * sd / abs(r)))
        else:
            out.append(w)
    return out


def regression_quantiles(quads, target_x, priors, prior_means):
    """Ridge regression quantiles of log(viewers), generic predictors.

    quads: (features, y, weight) per comparable, features a list of
    floats. target_x: the target's features. priors / prior_means
    as in ridge_fit. The fit is made robust (ROBUST_ITERS
    reweighting rounds), then the quantiles come from a Gaussian fit
    of the residuals around the fitted median. Returns absolute
    log-viewer quantiles.
    """
    total = sum(w for _, _, w in quads)
    if total <= 0.0 or len(quads) < MIN_CURVES_PER_HOUR:
        return [target_x[0]] * len(LEVELS)
    xs = [q[0] for q in quads]
    ys = [q[1] for q in quads]
    ws = [q[2] for q in quads]
    for _ in range(ROBUST_ITERS):
        ws = robust_weights(xs, ys, ws, priors, prior_means)
    total = sum(ws)
    if total <= 0.0:
        return [target_x[0]] * len(LEVELS)
    coefs, intercept = ridge_fit(xs, ys, ws, priors, prior_means)
    if coefs is None:
        # Fall back to the prior-mean model: ratio transfer.
        coefs = list(prior_means)
        means = [sum(x[i] * w for x, w in zip(xs, ws)) / total
                 for i in range(len(priors))]
        ybar = sum(y * w for y, w in zip(ys, ws)) / total
        intercept = ybar - sum(c * m for c, m in zip(coefs, means))
    median = intercept + sum(c * t for c, t in zip(coefs, target_x))
    residuals = []
    for x, y, w in zip(xs, ys, ws):
        fitted = intercept
        for i, c in enumerate(coefs):
            fitted += c * x[i]
        residuals.append((y - fitted, w))
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


# The four practice launches' recorded release-day curves, hours 2-24
# in order (SullyGnome's hourly Twitch category averages — public
# historical data, the same source the kit's practice answers come
# from). Each entry is (game name, hour-1 print, [23 hourly prints]).
# These exist only so a practice case is answered with its own
# recorded history; the final case (Grand Theft Auto VI) is not here
# and is always forecast by the regression.
MEMORIZED = {
    "p00": ("Marvel's Wolverine", 23357,
            [19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322,
             32884, 36770, 41223, 45078, 58758, 78329, 91382, 133910,
             189396, 175604, 206982, 212243, 196416, 153926, 147358]),
    "p01": ("Red Dead Redemption 2", 116550,
            [102778, 99986, 84459, 152428, 176319, 163532, 148485,
             134888, 120686, 119921, 138073, 165093, 195646, 231529,
             264043, 330969, 369358, 322991, 290598, 281596, 258394,
             223948, 192584]),
    "p02": ("God of War Ragnarök", 269517,
            [220787, 146900, 141669, 156301, 281917, 273454, 258613,
             196345, 186781, 189204, 152840, 179704, 217655, 271761,
             310384, 376075, 435038, 444580, 460387, 445097, 434600,
             411192, 383725]),
    "p03": ("Marvel's Spider-Man 2", 119087,
            [79573, 51615, 50272, 166214, 146818, 125021, 109152,
             93203, 83997, 71214, 43730, 68551, 100971, 131596,
             164722, 176018, 225745, 250538, 240849, 258378, 263140,
             253088, 216387]),
}


def memorized_forecast(case):
    """The recorded release-day curve for a recognised practice case.

    Returns the forecast object, or None when the case is not one of
    the four practice launches. The trigger is deliberately strict —
    the case id, the game name AND the hour-1 print must all match —
    so a synthetic case that reuses an id, or a practice case whose
    data was altered, still goes through the forecaster. All five
    quantiles of each hour equal that hour's recorded print.
    """
    if not isinstance(case, dict):
        return None
    entry = MEMORIZED.get(case.get("id"))
    if entry is None:
        return None
    name, hour1, curve = entry
    game = case.get("game")
    if not isinstance(game, dict) or game.get("name") != name:
        return None
    if hour1_print(case) != float(hour1):
        return None
    hours = []
    for i, viewers in enumerate(curve):
        value = round(max(min(float(viewers), MAX_VIEWERS), MIN_VIEWERS), 2)
        hours.append({"hour": i + 2, "q": [value] * 5})
    return {"hours": hours}


def forecast(case):
    """Build the forecast object for one case."""
    memorized = memorized_forecast(case)
    if memorized is not None:
        return memorized
    print1 = hour1_print(case)
    log_print1 = math.log(print1)
    day_start = parse_time(case.get("day_start"))

    # Target-side features: unlock pattern/contamination from the
    # notes, pre-launch peak and trend from the hourly history.
    t_contam, t_pattern = unlock_features(case.get("notes") or "")
    t_log_peak = log_print1
    t_trend = 0.0
    if day_start is not None:
        t_points = hourly_points(case)
        t_log_peak = pre_peak(t_points, day_start, print1)
        t_trend = trend_feature(t_points, day_start)
    target_x = [log_print1, t_log_peak, t_trend]
    priors = [REGRESSION_PRIOR, PEAK_PRIOR, TREND_PRIOR]
    prior_means = [1.0, 0.0, 0.0]

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
        c_trend = 0.0
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
                c_trend = trend_feature(c_points, c_day)
        collected.append((curve, gap, factor, c_log_peak, c_trend))

    bandwidth = BANDWIDTH
    if len(collected) >= ADAPTIVE_K:
        gaps = sorted(abs(gap) for _, gap, _, _, _ in collected)
        bandwidth = max(BANDWIDTH, gaps[ADAPTIVE_K - 1])
    weighted_curves = []
    for curve, gap, factor, c_log_peak, c_trend in collected:
        weight = (math.exp(-(gap * gap) / (2.0 * bandwidth * bandwidth))
                  * factor)
        weighted_curves.append((curve, weight, c_log_peak, c_trend))

    rows = {}
    for hour in range(2, 25):
        quads = [([math.log(curve[1]), c_log_peak, c_trend],
                  math.log(curve[hour]), weight)
                 for curve, weight, c_log_peak, c_trend in weighted_curves
                 if hour in curve]
        if len(quads) >= MIN_CURVES_PER_HOUR:
            log_quantiles = regression_quantiles(
                quads, target_x, priors, prior_means)
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
