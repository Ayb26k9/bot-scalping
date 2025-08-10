
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time

# === CONFIG ===
SYMBOLS_FIXED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]
LIMIT = 100

# Telegram config
TELEGRAM_BOT_TOKEN = "TON_TOKEN_TELEGRAM_ICI"
TELEGRAM_CHAT_ID = "TON_CHAT_ID_TELEGRAM_ICI"
TEST_MODE = True  # True = affiche en console, False = envoie sur Telegram

# Strategy params
EMA_FAST = 7
EMA_SLOW = 25
RSI_WINDOW = 7
ADX_WINDOW = 14
BB_WINDOW = 20
VOL_WINDOW = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_THRESHOLD = 25
RSI_BUY_MIN, RSI_BUY_MAX = 50, 65
RSI_SELL_MIN, RSI_SELL_MAX = 35, 50

SLEEP_BETWEEN_CALLS = 0.15

# === FUNCTIONS ===
def fetch_binance_klines(symbol, interval, limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def compute_indicators(df):
    df["EMA_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=RSI_WINDOW - 1, adjust=False).mean()
    ma_down = down.ewm(com=RSI_WINDOW - 1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"].fillna(50, inplace=True)

    df["tr0"] = df["high"] - df["low"]
    df["tr1"] = (df["high"] - df["close"].shift()).abs()
    df["tr2"] = (df["low"] - df["close"].shift()).abs()
    df["TR"] = df[["tr0", "tr1", "tr2"]].max(axis=1)
    df["+dm"] = np.where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]),
                         np.maximum(df["high"] - df["high"].shift(), 0), 0)
    df["-dm"] = np.where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()),
                         np.maximum(df["low"].shift() - df["low"], 0), 0)
    df["TR_s"] = df["TR"].rolling(window=ADX_WINDOW).sum()
    df["+dm_s"] = df["+dm"].rolling(window=ADX_WINDOW).sum()
    df["-dm_s"] = df["-dm"].rolling(window=ADX_WINDOW).sum()
    df["+di"] = 100 * (df["+dm_s"] / df["TR_s"].replace(0, np.nan))
    df["-di"] = 100 * (df["-dm_s"] / df["TR_s"].replace(0, np.nan))
    df["DX"] = 100 * (df["+di"] - df["-di"]).abs() / (df["+di"] + df["-di"]).replace(0, np.nan)
    df["ADX"] = df["DX"].rolling(window=ADX_WINDOW).mean().fillna(0)

    sma = df["close"].rolling(window=BB_WINDOW).mean()
    std = df["close"].rolling(window=BB_WINDOW).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std

    df["vol_ma"] = df["volume"].rolling(window=VOL_WINDOW).mean().fillna(method="bfill")
    return df

def evaluate_conditions(latest):
    ema_buy = latest["EMA_fast"] > latest["EMA_slow"]
    ema_sell = latest["EMA_fast"] < latest["EMA_slow"]

    macd_buy = latest["MACD"] > latest["MACD_signal"]
    macd_sell = latest["MACD"] < latest["MACD_signal"]

    rsi_buy = RSI_BUY_MIN <= latest["RSI"] <= RSI_BUY_MAX
    rsi_sell = RSI_SELL_MIN <= latest["RSI"] <= RSI_SELL_MAX

    adx_ok = latest["ADX"] >= ADX_THRESHOLD
    vol_ok = latest["volume"] >= latest["vol_ma"]

    bb_buy = latest["close"] > latest["bb_upper"]
    bb_sell = latest["close"] < latest["bb_lower"]

    if all([ema_buy, macd_buy, rsi_buy, adx_ok, vol_ok, bb_buy]):
        return "BUY"
    if all([ema_sell, macd_sell, rsi_sell, adx_ok, vol_ok, bb_sell]):
        return "SELL"
    return "NEUTRAL"

def analyze_symbol(symbol):
    signals = []
    per_tf = {}
    for tf in TIMEFRAMES:
        time.sleep(SLEEP_BETWEEN_CALLS)
        df = fetch_binance_klines(symbol, tf, LIMIT)
        df = compute_indicators(df)
        latest = df.iloc[-1]
        signal = evaluate_conditions(latest)
        per_tf[tf] = signal
        signals.append(signal)
    if all(s == "BUY" for s in signals):
        return "BUY", per_tf
    elif all(s == "SELL" for s in signals):
        return "SELL", per_tf
    else:
        return "NEUTRAL", per_tf

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    requests.post(url, data=payload)

if __name__ == "__main__":
    while True:
        for symbol in SYMBOLS_FIXED:
            consensus, details = analyze_symbol(symbol)
            msg = f"📊 {symbol}\nSignal global : {consensus}\nDétails : {details}"
            if TEST_MODE:
                print(msg)
            else:
                send_telegram_message(msg)
        time.sleep(60)
