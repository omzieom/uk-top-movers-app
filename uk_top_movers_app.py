import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

# ---------- TICKER HELPERS ----------

def _extract_tickers_from_tables(tables):
    """
    Try to find a column that looks like UK tickers:
    short strings (<6 chars), mostly uppercase, no spaces.
    """
    for table in tables:
        for col in table.columns:
            series = table[col].astype(str).str.strip()
            if series.empty:
                continue
            max_len = series.str.len().max()
            if max_len > 6:
                continue
            # simple heuristic: majority uppercase & no spaces
            sample = series.head(20)
            no_space_ratio = (sample.str.contains(" ").sum()) / len(sample)
            # if mostly no spaces and not too long, call it tickers
            if no_space_ratio < 0.3:
                tickers = series.tolist()
                return [t for t in tickers if t and t != "nan"]
    return []


@st.cache_data(show_spinner=False)
def get_ftse100_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    tickers = _extract_tickers_from_tables(tables)
    return [t + ".L" for t in tickers]


@st.cache_data(show_spinner=False)
def get_ftse250_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    tickers = _extract_tickers_from_tables(tables)
    return [t + ".L" for t in tickers]


@st.cache_data(show_spinner=False)
def get_aim100_tickers():
    # reasonably liquid subset: FTSE AIM 100
    url = "https://en.wikipedia.org/wiki/FTSE_AIM_100_Index"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    tickers = _extract_tickers_from_tables(tables)
    return [t + ".L" for t in tickers]


@st.cache_data(show_spinner=False)
def get_universe(universe: str):
    ftse100 = get_ftse100_tickers()
    ftse250 = get_ftse250_tickers()
    aim100 = get_aim100_tickers()

    if universe == "FTSE 100":
        tickers = ftse100
    elif universe == "FTSE 250":
        tickers = ftse250
    elif universe == "FTSE 350":
        tickers = list(sorted(set(ftse100 + ftse250)))
    elif universe == "AIM 100":
        tickers = aim100
    elif universe == "All UK":
        tickers = list(sorted(set(ftse100 + ftse250 + aim100)))
    else:
        tickers = ftse100

    # de-duplicate but keep order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

# ---------- INDICATORS & QUANT-STYLE SCORING ----------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Returns & volatility
    returns = close.pct_change()
    df["ret"] = returns
    df["vol_20"] = returns.rolling(20).std()
    df["vol_60"] = returns.rolling(60).std()

    # Time-series momentum (20d & 60d), volatility-scaled (Sharpe-like)
    df["mom_20"] = close.pct_change(20)
    df["mom_60"] = close.pct_change(60)
    df["ts_mom_20"] = df["mom_20"] / (df["vol_20"] * np.sqrt(20))
    df["ts_mom_60"] = df["mom_60"] / (df["vol_60"] * np.sqrt(60))

    # Donchian-style breakout (55-day)
    high55 = close.rolling(55).max()
    low55 = close.rolling(55).min()
    df["breakout_up"] = (close - high55) / high55
    df["breakout_down"] = (close - low55) / low55

    # Trend filter (50 & 200 day MAs)
    df["ma_50"] = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()

    # ATR (14) - volatility range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["atr_pct"] = df["ATR"] / close

    # RSI (14) for overbought/oversold context
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def score_latest(df: pd.DataFrame):
    # Need enough history for the longest window
    if df.shape[0] < 220:
        return None

    latest = df.iloc[-1]

    ts_mom_20 = latest["ts_mom_20"]
    ts_mom_60 = latest["ts_mom_60"]
    breakout_up = latest["breakout_up"]
    breakout_down = latest["breakout_down"]
    ma50 = latest["ma_50"]
    ma200 = latest["ma_200"]
    price = latest["Close"]
    atr_pct = latest["atr_pct"]
    rsi = latest["RSI"]

    if any(pd.isna(x) for x in [ts_mom_20, ts_mom_60, ma50, ma200, atr_pct, rsi]):
        return None

    # Trend regime: +1 uptrend, -1 downtrend, 0 neutral
    if price > ma50 > ma200:
        trend_regime = 1
    elif price < ma50 < ma200:
        trend_regime = -1
    else:
        trend_regime = 0

    # Core signal: volatility-scaled momentum + breakout, aligned with trend
    core_signal = (
        0.6 * ts_mom_20 +
        0.4 * ts_mom_60 +
        1.0 * breakout_up -
        1.0 * breakout_down
    ) * (1 if trend_regime != 0 else 0.5)

    # Guardrails: if RSI extreme, dampen signal
    if rsi > 80 or rsi < 20:
        core_signal *= 0.7

    # Direction & expected move
    if core_signal > 0:
        direction = "BUY"
    else:
        direction = "SELL"

    expected_abs_move = float(abs(core_signal) * 5.0)  # heuristic scaling to %
    # Timeframe: higher ATR/vol → shorter horizon
    if atr_pct > 0.03 or df["vol_20"].iloc[-1] > 0.025:
        timeframe = "short (1–7 days)"
    else:
        timeframe = "medium (1–4 weeks)"

    # Confidence: squash |signal| into (0,1)
    confidence = float(np.tanh(abs(core_signal)))

    return {
        "price": float(price),
        "expected_movement_pct": round(expected_abs_move, 2),
        "direction": direction,
        "timeframe": timeframe,
        "confidence": round(confidence, 2),
    }

# ---------- SCAN UNIVERSE ----------

@st.cache_data(show_spinner=True)
def analyse_universe(tickers, period="1y", top_n=10, batch_size=30):
    results = []
    tickers = list(dict.fromkeys(tickers))  # de-duplicate

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        st.write(
            f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Downloading batch {i}-{i+len(batch)-1} ({len(batch)} tickers)…"
        )
        try:
            data = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
            except Exception:
                continue

            if df is None or df.empty:
                continue

            df = compute_indicators(df)
            scored = score_latest(df)
            if not scored:
                continue

            results.append(
                {
                    "ticker": t,
                    "close": scored["price"],
                    "expected_movement_pct": scored["expected_movement_pct"],
                    "direction": scored["direction"],
                    "timeframe": scored["timeframe"],
                    "confidence": scored["confidence"],
                }
            )

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    df_res["rank_key"] = df_res["expected_movement_pct"] * df_res["confidence"]
    df_res = df_res.sort_values(by="rank_key", ascending=False).head(top_n)
    return df_res.drop(columns=["rank_key"])

# ---------- STREAMLIT UI ----------

def main():
    st.title("UK Top Movers Scanner 🇬🇧")
    st.markdown(
        "Quant-style heuristic scanner for UK stocks (FTSE 100 / 250 / 350 / AIM 100).  "
        "Uses volatility-scaled momentum, breakouts, and trend filters to highlight "
        "candidates likely to move.  **Not financial advice.**"
    )

    universe = st.selectbox(
        "Universe",
        ["FTSE 100", "FTSE 250", "FTSE 350", "AIM 100", "All UK"],
        index=4,
    )

    top_n = st.slider("Number of top movers to show", 5, 50, 10, step=5)

    period = st.selectbox(
        "Lookback period",
        ["6mo", "1y", "2y"],
        index=1,
        help="Longer lookback improves trend filters but increases load time.",
    )

    if st.button("Run scan"):
        with st.spinner("Fetching tickers…"):
            tickers = get_universe(universe)
        st.write(f"Analysing **{len(tickers)}** tickers. This can take a few minutes.")

        df_res = analyse_universe(tickers, period=period, top_n=top_n)

        if df_res.empty:
            st.error("No results. Check connection and try again.")
            return

        st.subheader("Top predicted movers")
        st.dataframe(
            df_res.style.format(
                {
                    "close": "{:.2f}",
                    "expected_movement_pct": "{:.2f}",
                    "confidence": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            csv,
            "uk_top_movers_results.csv",
            "text/csv",
        )

if __name__ == "__main__":
    main()
