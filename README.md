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
about 0.1 seconds per case, and its two runs on the same case are
byte-identical.

## Approach: launch-day shape transfer

A launch's hour-1 print is the strongest single signal of its day-shape,
because it measures how much of the launch had already happened by 01:00 UTC
(a huge hour 1 means the launch peaked early; a tiny one means the unlock
still lies ahead). The forecaster therefore:

1. Takes every comparable launch in the case and computes, per hour, the
   log-ratio of that hour's count to the launch's own hour-1 count.
2. Weights each comparable by how close its hour-1 print is to this case's
   (Gaussian kernel in log space, bandwidth 0.9 nats), down-weighting
   early-access-contaminated launches (×0.5), whose day-1 curves are
   flattened by players already in-game before launch.
3. Anchors on this case's hour-1 print and applies the weighted empirical
   quantiles of the log-ratios per hour, lightly smoothed across
   neighbouring hours, so the spread is calibrated from real launch
   dispersion rather than guessed.

It never crashes on malformed or sparse input: any failure degrades to a
valid flat forecast, and hours with too few comparables fall back to a
flat forecast with a widening spread.

## Local validation

Scored locally with an independent pinball-loss scorer against the four
practice cases' actual release-day counts (Marvel's Wolverine, Red Dead
Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2):

| case | this entry | house baseline |
| --- | --- | --- |
| p00 Wolverine | 0.336 | 0.744 |
| p01 Red Dead Redemption 2 | 0.206 | 0.287 |
| p02 God of War Ragnarök | 0.141 | 0.215 |
| p03 Spider-Man 2 | 0.196 | 0.303 |
| **mean** | **0.220** | **0.387** |

43% lower mean pinball loss than the house baseline (whose published
practice loss of ~0.387 the local scorer reproduces at 0.3871). A
leave-one-out backtest over the 37 comparable launch curves gives a mean
loss of 0.40 against the house baseline's 0.93 on the same targets.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes one `<id>.out` per `<id>.in` found in the input directory.

## License

MIT — see [LICENSE](LICENSE).
