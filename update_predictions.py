import gc
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import BDay
from vnstock import Vnstock

from model import KronosTokenizer, Kronos, KronosPredictor

# --- Configuration ---
# MODEL_PATH is overridable so CI (GitHub Actions) can point it at a cache-friendly
# path inside the workspace; locally it defaults to the sibling `../Kronos_model` dir.
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": os.environ.get("KRONOS_MODEL_PATH", "../Kronos_model"),
    "SYMBOL": 'BMP',
    "EXCHANGE": 'HOSE',
    "DATA_SOURCE": 'KBS',
    "INTERVAL": '1D',
    "HIST_POINTS": 360,     # daily bars (~1.5 years of trading days)
    "PRED_HORIZON": 10,     # trading-day forecast horizon
    "N_PREDICTIONS": 30,
    "VOL_WINDOW": 20,       # ~1 month of trading days for baseline vol
}


def load_model():
    """Loads the Kronos model and tokenizer."""
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k", cache_dir=Config["MODEL_PATH"])
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini", cache_dir=Config["MODEL_PATH"])
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print("Model loaded successfully.")
    return predictor


def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + BDay(1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq='B'
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        close_preds_main, volume_preds_main = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

        # print("Making volatility prediction (T=0.9)...")
        # begin_time = time.time()
        # close_preds_volatility, _ = predictor.predict(
        #     df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        #     pred_len=Config["PRED_HORIZON"], T=0.9, top_p=0.9,
        #     sample_count=Config["N_PREDICTIONS"], verbose=True
        # )
        # print(f"Volatility prediction completed in {time.time() - begin_time:.2f} seconds.")
        close_preds_volatility = close_preds_main

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_vnstock_data():
    """Fetches daily OHLCV data for a HOSE ticker via vnstock."""
    symbol = Config["SYMBOL"]
    needed = Config["HIST_POINTS"] + Config["VOL_WINDOW"]
    # Calendar buffer: ~1.55x to cover weekends/holidays, plus 30d headroom.
    lookback_calendar_days = int(needed * 1.55) + 30

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_calendar_days)

    print(f"Fetching {symbol} daily bars from {start_date} to {end_date} via vnstock ({Config['DATA_SOURCE']})...")
    stock = Vnstock().stock(symbol=symbol, source=Config["DATA_SOURCE"])
    df = stock.quote.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1D',
    )

    df = df.rename(columns={'time': 'timestamps'})
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    # vnstock does not return a quote volume; synthesize "amount" as volume * typical price
    # so the Kronos predictor's amount feature is populated consistently with Binance-style data.
    df['amount'] = df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)

    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df.sort_values('timestamps').reset_index(drop=True)
    df = df.tail(needed).reset_index(drop=True)

    print(f"Data fetched successfully ({len(df)} bars, latest {df['timestamps'].iloc[-1].date()}).")
    return df


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """Calculates upside and volatility amplification probabilities over the forecast horizon."""
    last_close = hist_df['close'].iloc[-1]

    final_bar_preds = close_preds_df.iloc[-1]
    upside_prob = (final_bar_preds > last_close).mean()

    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    horizon = Config["PRED_HORIZON"]
    print(f"Upside Probability ({horizon}d): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df):
    """Generates and saves a comprehensive forecast chart."""
    print("Generating comprehensive forecast chart...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    pred_time = pd.to_datetime(close_preds_df.index)

    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{Config["SYMBOL"]} ({Config["EXCHANGE"]}) Probabilistic Price & Volume Forecast (Next {Config["PRED_HORIZON"]} Trading Days)', fontsize=16, weight='bold')
    ax1.set_ylabel('Price (VND)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.8)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=0.8)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = hist_time.iloc[-1] + timedelta(hours=12)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / 'prediction_chart.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


def update_html(upside_prob, vol_amp_prob):
    """
    Updates the index.html file with the latest metrics and timestamp.
    This version uses a more robust lambda function for replacement to avoid formatting errors.
    """
    print("Updating index.html...")
    html_path = Config["REPO_PATH"] / 'index.html'
    now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    upside_prob_str = f'{upside_prob:.1%}'
    vol_amp_prob_str = f'{vol_amp_prob:.1%}'

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Robustly replace content using lambda functions
    content = re.sub(
        r'(<strong id="update-time">).*?(</strong>)',
        lambda m: f'{m.group(1)}{now_utc_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="upside-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{upside_prob_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="vol-amp-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{vol_amp_prob_str}{m.group(2)}',
        content
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML file updated successfully.")


def main_task(model):
    """Executes one full update cycle.

    Pure compute: fetches data, runs the model, rewrites `index.html` and
    `prediction_chart.png`. Commit + deploy are handled by the GitHub Actions
    workflow (`.github/workflows/forecast.yml`), not by this script.
    """
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    df_full = fetch_vnstock_data()

    close_preds, volume_preds, v_close_preds = make_prediction(df_full, model)

    hist_df_for_plot = df_full.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_full.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    create_plot(hist_df_for_plot, close_preds, volume_preds)
    update_html(upside_prob, vol_amp_prob)

    del df_full, close_preds, volume_preds, v_close_preds
    del hist_df_for_plot, hist_df_for_metrics
    gc.collect()

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model()
    main_task(loaded_model)