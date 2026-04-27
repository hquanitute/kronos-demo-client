# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A self-updating demo site for the **Kronos** financial foundation model. `update_predictions.py` fetches daily K-line data for **multiple HOSE symbols** (`BMP`, `D2D`, `LPB`, `FPT`, `TCB`) via the `vnstock` library, runs a Monte-Carlo forecast with `Kronos-mini` for each, and re-renders per-symbol `prediction_chart_{SYMBOL}.png` files and `index.html` with fresh metrics. **The script itself no longer schedules or deploys** — `.github/workflows/forecast.yml` runs it daily on GitHub Actions and publishes the output via the GitHub Pages artifact pipeline (`actions/upload-pages-artifact` → `actions/deploy-pages`). No forecast commits land on `master` anymore; history stays clean.

## Run it

Dependencies are managed with **uv** — there is a `pyproject.toml` + `uv.lock`, no `requirements.txt`. Use `uv` for everything Python:

```bash
uv sync                              # install locked deps
uv run python update_predictions.py  # run a single forecast cycle locally
uv add <pkg>                         # add a new dep (never `pip install`)
```

`update_predictions.py` now runs **exactly one update cycle and exits** — all scheduling lives in `.github/workflows/forecast.yml` (`cron: "0 1 * * *"` = 01:00 UTC = 08:00 ICT, pre-market). Running locally just regenerates the static files in place; it won't commit or push. To trigger a CI deploy on demand, use `workflow_dispatch` from the Actions tab or `gh workflow run forecast.yml`. There are no tests, linters, or build steps.

## Non-obvious gotchas

- **`MODEL_PATH` is env-overridable.** Locally defaults to the *sibling* `../Kronos_model` (outside the repo); CI overrides via `KRONOS_MODEL_PATH=./.kronos_model_cache` so `actions/cache` can persist HuggingFace weights (`NeoQuasar/Kronos-Tokenizer-2k`, `NeoQuasar/Kronos-mini`) between runs. If you change the pinned model IDs, bump the cache `key` suffix in `forecast.yml` or the workflow will keep restoring stale weights.
- **The static site is deployed via the Pages artifact job**, not by committing back to `master`. The `build` job stages `index.html`, `style.css`, `prediction_chart.png`, and `img/` into `_site/`, uploads it, and the `deploy` job publishes. **If you add a new static asset, add it to the `Stage static site` step in `forecast.yml` or it won't be served.**
- **`index.html` is edited by regex**, not a template engine. The update regexes in `update_html()` target one shared anchor (`<strong id="update-time">`) plus per-symbol anchors `<p class="metric-value" id="upside-prob-{SYMBOL}">` and `<p class="metric-value" id="vol-amp-prob-{SYMBOL}">`. Preserve those IDs verbatim (with the symbol suffix) or daily updates will silently no-op for that symbol.
- **No last-bar drop here.** Unlike the original Binance hourly variant, we do *not* `iloc[:-1]` the dataframe, because the CI job runs pre-market (01:00 UTC / 08:00 ICT) and vnstock's latest row is already a complete previous-session close.
- **`vnstock` has no quote-volume field**, so `fetch_vnstock_data()` synthesizes an `amount` column as `volume × mean(OHLC)` to feed the Kronos predictor the same 6-feature vector (`open,high,low,close,volume,amount`) the model was trained on. Don't remove this.
- **`DATA_SOURCE` is a vnstock backend identifier** (e.g. `VCI`, `TCBS`, `KBS`, `SSI`). Swap it in `Config` if the current provider rate-limits or returns stale data — the rest of the fetch code is provider-agnostic.
- **Forecast timestamps use pandas business-day frequency** (`BDay(1)` + `freq='B'`), which skips weekends but **not Vietnamese public holidays**. The x-axis labels on forecast bars may be off by a day or two across Lunar New Year, Reunification Day, etc. — acceptable for a demo, not for trading.
- **Device is hard-coded to CPU** (`device="cpu"` in `load_model`). Change with care; the rest of the script assumes CPU timings.
- **Volatility "amplification" prediction reuses the main forecast** (see the commented-out second `predictor.predict` call and `close_preds_volatility = close_preds_main`). The "two-temperature" methodology described in the HTML is currently a single run.

## Code layout

- `update_predictions.py` — config, data fetch, prediction, metrics, plot, HTML patch. `main_task()` loops over all `SYMBOLS`, running the full pipeline per symbol, then calls `update_html()` once with all results. One call from `__main__`, exits when done. No scheduler, no git calls.
- `.github/workflows/forecast.yml` — cron + `workflow_dispatch` trigger, `uv sync` + `uv run python update_predictions.py`, then stage `_site/` and deploy via `actions/deploy-pages@v4`.
- `model/` — vendored copy of the Kronos model code. `kronos.py` defines `KronosTokenizer`, `Kronos` (both `PyTorchModelHubMixin` — loaded via `from_pretrained`), and `KronosPredictor` (the inference wrapper used by `update_predictions.py`). `module.py` holds the transformer blocks and `BSQuantizer`. Treat this directory as upstream code — prefer not to edit it unless syncing from the Kronos repo.
- `index.html`, `style.css`, `img/logo.png`, `prediction_chart_{SYMBOL}.png` (one per symbol) — the static site served by GitHub Pages.

## How Kronos inference works

`make_prediction()` → `KronosPredictor.predict()` → `auto_regressive_inference()` in `model/kronos.py`.

**Step 1 — normalize.** The 6-feature input matrix (`open,high,low,close,volume,amount`, shape `[hist_points, 6]`) is z-scored column-wise using its own mean/std, then clipped to ±5 σ.

**Step 2 — tokenize (encode).** `KronosTokenizer` runs an encoder Transformer that compresses the input into a two-level token sequence: a coarse `s1` token and a fine `s2` token conditioned on `s1`.

**Step 3 — autoregressive decoding (Monte-Carlo sampling).** The context is replicated `N_PREDICTIONS` (30) times in a single batched forward pass. For each of the `PRED_HORIZON` (10) forecast steps:
1. `model.decode_s1(tokens, stamp)` produces `s1` logits → sampled via `top_p=0.95` nucleus sampling at `T=1.0`.
2. `model.decode_s2(context, s1_sample)` produces `s2` logits → sampled the same way.
3. The new `(s1, s2)` token pair is appended to the token sequence; repeat.

Time-stamp features (`minute, hour, weekday, day, month`) are passed as a separate `x_stamp` / `y_stamp` tensor alongside the token sequence.

**Step 4 — decode & de-normalize.** The final token sequence is decoded back to the feature space by `tokenizer.decode()`, clipped to the last `pred_len` steps, then de-normalized with the original mean/std to recover price-scale values.

**Step 5 — extract outputs.** Column index 3 (close) and 4 (volume) are sliced from the decoded tensor. The result is a `DataFrame[pred_horizon × n_predictions]` — 30 independent close-price trajectories over 10 trading days.

**What the metrics use.** `calculate_metrics()` computes:
- *Upside probability*: fraction of paths whose final-bar close > last known close.
- *Volatility amplification probability*: fraction of paths whose realized log-return stdev exceeds the `VOL_WINDOW`-day historical stdev.

The "volatility" paths currently reuse the main prediction (`close_preds_volatility = close_preds_main`) — the commented-out second `predictor.predict` call at `T=0.9` would have been a separate lower-temperature sample.

## Forecast config

Tuning knobs live in the `Config` dict at the top of `update_predictions.py`:

- `SYMBOLS` — list of HOSE tickers to forecast; currently `['BMP', 'D2D', 'LPB', 'FPT', 'TCB']`. Add or remove symbols here; the loop in `main_task()` and the regex anchors in `index.html` must stay in sync.
- `EXCHANGE` / `DATA_SOURCE` — `HOSE` / `KBS` via vnstock.
- `INTERVAL='1D'` — daily bars.
- `HIST_POINTS=360` — ~1.5 years of daily context fed to the model.
- `PRED_HORIZON=10` — trading-day forecast horizon.
- `N_PREDICTIONS=30` — Monte-Carlo sample paths.
- `VOL_WINDOW=20` — ~1 month of daily bars used as the baseline-volatility reference.

The cadence itself (01:00 UTC daily) lives in `.github/workflows/forecast.yml` (`cron: "0 1 * * *"`). To change it, edit the cron expression there — do *not* add it back to `Config`.

The two headline metrics in `calculate_metrics` are **upside probability** (share of sampled paths whose final-bar close exceeds the last known close) and **volatility amplification probability** (share of paths whose realized log-return stdev exceeds recent historical stdev).
