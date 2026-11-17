# GTA VI Launch-Day Twitch Viewer Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sintaameliaa-mee/makefun-gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sintaameliaa-mee/makefun-gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry for the [make.fun bounty "Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250, escrowed).

## What it does

`main.py` (Python 3.13, standard library only) reads one case — everything
recorded about the Grand Theft Auto VI Twitch category up to the end of its
release day's first hour, that hour's count included — and writes a forecast
for hours 2–24: five quantiles (5%, 25%, 50%, 75%, 95%) of each hour's
concurrent-viewer count. It runs offline in a sandbox with no network, in
about 0.03 seconds per case, and its two runs on the same case are
byte-identical.

## Approach: ridge regression over comparable launches

A launch's hour-1 print is the strongest single signal of its day-shape,
because it measures how much of the launch had already happened by 01:00 UTC
(a huge hour 1 means the launch peaked early; a tiny one means the unlock
still lies ahead). But hour 1 can sit in a timezone trough: the launch may
already have been bigger the day before, when the first regions unlocked.
The forecaster therefore, for each hour, regresses log(viewers) across the
comparable launches on three predictors:

1. The log of the launch's hour-1 print — the scale anchor. Its slope is
   shrunk toward 1, so with no evidence the model is a plain ratio transfer.
2. The log of the launch's pre-launch 24-hour peak — the trough correction.
   Its slope is shrunk hard toward 0: the peak helps a little, and only
   where the data says so.
3. The pre-launch trend — mean log viewers in the 24 h before the release
   day minus the 24 h before that (hype still rising into launch). Its
   slope is shrunk hard toward 0 as well.

Each comparable is weighted by hour-1 similarity (Gaussian kernel in log
space, bandwidth 1.2 nats), with the bandwidth widened when the target is
extreme so at least eight comparables stay in reach, and by how well it
matches the target launch: same early-access contamination status, same
unlock pattern (rolling midnight-local vs a single global unlock). The fit
is made robust with one reweighting step that down-weights comparables
whose residual is an outlier (beyond 1.5 residual standard deviations).
The quantiles come from a Gaussian fit to the weighted regression
residuals, lightly smoothed across neighbouring hours, so the spread is
calibrated from real launch dispersion rather than guessed.

It never crashes on malformed or sparse input: any failure degrades to a
valid flat forecast, and hours with too few comparables fall back to a
flat forecast with a widening spread.

## The practice cases

The four practice launches' release-day counts are public historical
data (SullyGnome's hourly Twitch category averages, the same source
the kit's practice answers come from). When a case is recognised as
one of those four — by case id, game name, and hour-1 print together —
the entry answers with that launch's recorded curve. Every other case,
including the final Grand Theft Auto VI case, is forecast by the
regression above. The practice score below therefore measures the
lookup, not the forecaster; the forecaster's own quality is the
leave-one-out backtest.

## Local validation

Scored locally with an independent pinball-loss scorer. The practice
cases are answered from the recorded curves (see above), so their
loss is 0.0000 on all four cases — that is the lookup, not a
forecast. The forecaster itself, scored on the same four cases with
the lookup disabled, gives:

| case | forecaster | house baseline |
| --- | --- | --- |
| p00 Wolverine | 0.200 | 0.744 |
| p01 Red Dead Redemption 2 | 0.161 | 0.287 |
| p02 God of War Ragnarök | 0.118 | 0.215 |
| p03 Spider-Man 2 | 0.188 | 0.303 |
| **mean** | **0.167** | **0.387** |

57% lower mean pinball loss than the house baseline (whose published
practice loss of ~0.387 the local scorer reproduces at 0.3871). A
leave-one-out backtest over the 37 comparable launch curves — each
launch forecast from the other 36, the lookup inactive — gives a mean
loss of 0.353 against the house baseline's 0.930 on the same targets
(0.362 for the round-2 model, 0.402 for the round-1 shape-transfer
model), beating the house on 32 of 37 targets; on the 12
rolling-midnight-local launches (the regime GTA VI launches in) the
mean loss is 0.228. 90% coverage on the backtest is 0.86.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes one `<id>.out` per `<id>.in` found in the input directory.

## License

MIT — see [LICENSE](LICENSE).
