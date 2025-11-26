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


def _get_tickers_from_wikipedia(url: str, min_count: int = 50) -> list:
    """
    Fetch a Wikipedia page and extract tickers from the
    constituents table that has a 'Ticker' / 'EPIC' / 'Symbol' column.
    """
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)

    candidate_names = ["ticker", "epic", "symbol", "code", "ric"]

    for table in tables:
        cols = [str(c) for c in table.columns]
        ticker_col_name = None
        for c in cols:
            lc = c.lower()
            if any(name in lc for name in candidate_names):
                ticker_col_name = c
                break
        if ticker_col_name is None:
            continue

        series = table[ticker_col_name].astype(str).str.strip()
        # filter out junk: short-ish, mostly no spaces
        series = series[series.str.len().between(1, 6)]
        series = series[~series.str.contains(" ", regex=False)]
        tickers = [t for t in series.tolist() if t and t.lower() not in ("ticker", "epic", "symbol")]
        if len(tickers) >= min_count:
            return tickers

    return []  # nothing suitable found


@st.cache_data(show_spinner=False)
def get_ftse100_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tickers = _get_tickers_from_wikipedia(url, min_count=80)
    return [t + ".L" for t in tickers]


@st.cache_data(show_spinner=False)
def get_ftse250_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    tickers = _get_tickers_from_wikipedia(url, min_count=200)
    return [t + ".L" for t in tickers]


def _get_aim_from_digitallook() -> list:
    """
    Try to pull AIM 100 constituents from DigitalLook.
    Returns [] on failure.
    """
    try:
        dl_url = (
            "https://www.digitallook.com/cgi-bin/dlmedia/security.cgi?"
            "security_classification_id=103317&country_id=1&trade_analysis=1&"
            "csi=113101&target_csi=&id=113101&sub_action=&orderby_field=security_name&"
            "price_type=closing_&intraday_prices=1&ac=&username=&action=constituents&"
            "selected_menu_link=%2Fdlmedia%2Finvesting&order_by=column7&"
            "view_data=standard&sequence=descending"
        )
        r = requests.get(dl_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        candidate_names = ["epic", "ticker", "symbol", "code"]

        for table in tables:
            cols = [str(c) for c in table.columns]
            epic_col = None
            for c in cols:
                lc = c.lower()
                if any(name in lc for name in candidate_names):
                    epic_col = c
                    break
            if epic_col is None:
                continue
            series = table[epic_col].astype(str).str.strip()
            series = series[series.str.len().between(1, 6)]
            series = series[~series.str.contains(" ", regex=False)]
            tickers = [t for t in series.tolist() if t and t.lower() not in ("epic", "ticker", "symbol")]
            if len(tickers) >= 40:  # sanity check
                return [t + ".L" for t in tickers]
    except Exception:
        pass

    return []


def _get_aim_from_yahoo_screener() -> list:
    """
    Best-effort fallback: use Yahoo Finance screener API to get UK/LSE small caps.
    This is NOT guaranteed to be strictly 'AIM 100', but gives a decent AIM-like universe.
    Fully wrapped in try/except; returns [] on any failure.
    """
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener"
        payload = {
            "size": 250,
            "offset": 0,
            "sortField": "intradaymarketcap",
            "sortType": "ASC",
            "quoteType": "EQUITY",
            "topOperator": "AND",
            "query": {
                "operator": "AND",
                "operands": [
                    {"operator": "EQ", "operands": ["GB", "region"]},
                    {"operator": "EQ", "operands": ["LSE", "exchange"]},
                    # Yahoo doesn’t have a clean “AIM index” flag publicly,
                    # but smaller caps tend to approximate AIM universe.
                    {"operator": "LT", "operands": [2_000_000_000, "marketcap"]},
                ],
            },
            "userId": "",
            "userIdType": "guid",
        }
        headers = dict(HEADERS)
        headers["Content-Type"] = "application/json"
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json()
        # structure: finance -> result[0] -> quotes
        quotes = (
            js.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
        symbols = [q.get("symbol") for q in quotes if isinstance(q, dict) and q.get("symbol")]
        # Keep only .L symbols to stay on LSE
        symbols = [s for s in symbols if s.endswith(".L")]
        return symbols
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_aim100_tickers() -> list:
    """
    AIM 100 or AIM-like universe:
    1) Try DigitalLook constituents table.
    2) If that fails or is too small, fallback to Yahoo screener.
    3) If both fail, return [] and show a warning in the UI.
    """
    tickers = _get_aim_from_digitallook()
    if len(tickers) >= 40:
        return tickers

    # fallback to Yahoo screener
    yf_tickers = _get_aim_from_yahoo_screener()
    if len(yf_tickers) >= 40:
        return yf_tickers

    st.warning(
        "AIM 100 / AIM-like tickers could not be reliably fetched from external sources. "
        "The AIM 100 and All UK universes will currently only include FTSE names."
    )
    return []


@st.cache_data(show_spinner=False)
def get_universe(universe: str) -> list:
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


# ---------- INDICATORS & RELAXED QUANT-STYLE SCORING ----------


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Returns & volatility
    returns = close.pct_change()
    df["ret"] = returns
    df["vol_20"] = returns.rolling(20, min_periods=10).std()
    df["vol_60"] = returns.rolling(60, min_periods=20).std()

    # Time-series momentum (20d & 60d), volatility-scaled (Sharpe-like)
    df["mom_20"] = close.pct_change(20)
    df["mom_60"] = close.pct_change(60)
    df["ts_mom_20"] = df["mom_20"] / (df["vol_20"] * np.sqrt(20))
    df["ts_mom_60"] = df["mom_60"] / (df["vol_60"] * np.sqrt(60))

    # Donchian-style breakout (55-day)
    high55 = close.rolling(55, min_periods=20).max()
    low55 = close.rolling(55, min_periods=20).min()
    df["breakout_up"] = (close - high55) / high55
    df["breakout_down"] = (close - low55) / low55

    # Trend filter (50 & 200 day MAs)
    df["ma_50"] = close.rolling(50, min_periods=25).mean()
    df["ma_200"] = close.rolling(200, min_periods=60).mean()

    # ATR (14) - volatility range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=7).mean()
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
    # Relaxed history requirement: ~160 trading days (~8 months)
    if df.shape[0] < 160:
        return None

    latest = df.iloc[-1]

    # Pull metrics and be forgiving with NaNs
    ts_mom_20 = latest.get("ts_mom_20", np.nan)
    ts_mom_60 = latest.get("ts_mom_60", np.nan)
    breakout_up = latest.get("breakout_up", np.nan)
    breakout_down = latest.get("breakout_down", np.nan)
    ma50 = latest.get("ma_50", np.nan)
    ma200 = latest.get("ma_200", np.nan)
    price = latest.get("Close", np.nan)
    atr_pct = latest.get("atr_pct", np.nan)
    rsi = latest.get("RSI", np.nan)

    # We absolutely need a price
    if pd.isna(price) or price <= 0:
        return None

    # Replace NaNs for non-critical metrics with 0 / neutral values
    for name in ("ts_mom_20", "ts_mom_60", "breakout_up", "breakout_down"):
        pass  # just a reminder in comments
    ts_mom_20 = 0.0 if pd.isna(ts_mom_20) else float(ts_mom_20)
    ts_mom_60 = 0.0 if pd.isna(ts_mom_60) else float(ts_mom_60)
    breakout_up = 0.0 if pd.isna(breakout_up) else float(breakout_up)
    breakout_down = 0.0 if pd.isna(breakout_down) else float(breakout_down)

    # Trend regime: use ma50/ma200 if available, else treat as neutral
    trend_regime = 0
    if not pd.isna(ma50) and not pd.isna(ma200):
        ma50 = float(ma50)
        ma200 = float(ma200)
        if price > ma50 > ma200:
            trend_regime = 1
        elif price < ma50 < ma200:
            trend_regime = -1

    # ATR pct: if missing, approximate with recent vol
    if pd.isna(atr_pct) or atr_pct <= 0:
        fallback_vol = df["ret"].rolling(20).std().iloc[-1]
        atr_pct = float(fallback_vol) if not pd.isna(fallback_vol) else 0.02
    else:
        atr_pct = float(atr_pct)

    # RSI: neutral if missing
    if pd.isna(rsi):
        rsi = 50.0
    else:
        rsi = float(rsi)

    # Core signal: volatility-scaled momentum + breakout
    core_signal = (
        0.6 * ts_mom_20 +
        0.4 * ts_mom_60 +
        1.0 * breakout_up -
        1.0 * breakout_down
    )

    # Trend weighting: boost when aligned, soften when neutral
    if trend_regime == 1 and core_signal > 0:
        core_signal *= 1.3
    elif trend_regime == -1 and core_signal < 0:
        core_signal *= 1.3
    elif trend_regime == 0:
        core_signal *= 0.9  # mild reduction, not a killer

    # RSI guardrails only for very extreme conditions
    if rsi > 85 or rsi < 15:
        core_signal *= 0.85

    # Direction & expected move (still heuristic)
    if core_signal >= 0:
        direction = "BUY"
    else:
        direction = "SELL"

    expected_abs_move = float(abs(core_signal) * 5.0)  # scaled to a % range

    # Timeframe: higher ATR/vol → shorter horizon
    vol_20 = df["vol_20"].iloc[-1]
    vol_20 = 0.0 if pd.isna(vol_20) else float(vol_20)
    if atr_pct > 0.03 or vol_20 > 0.025:
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
        batch = tickers[i:i + batch_size]
        st.write(
            f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Downloading batch {i}-{i + len(batch) - 1} ({len(batch)} tickers)…"
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

        # Handle both MultiIndex and simple column structures
        for t in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data.xs(t, level=1, axis=1)
                else:
                    if len(batch) == 1:
                        df = data
                    else:
                        if t not in data:
                            continue
                        df = data[t]
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
    st.title("UK Top Movers Scanner GB")
    st.markdown(
        "Quant-style heuristic scanner for UK stocks (FTSE 100 / 250 / 350 / AIM 100). "
        "Uses volatility-scaled momentum, breakouts, and trend filters to highlight "
        "candidates likely to move. **Not financial advice.**"
    )

    universe = st.selectbox(
        "Universe",
        ["FTSE 100", "FTSE 250", "FTSE 350", "AIM 100", "All UK"],
        index=4,
    )

    top_n = st.slider("Number of top movers to show", 5, 50, 10, step=5)

    period = st.selectbox(
        "Lookback period",
        ["1y", "2y"],
        index=0,
        help="Longer lookback improves trend filters but increases load time.",
    )

    if st.button("Run scan"):
        with st.spinner("Fetching tickers…"):
            tickers = get_universe(universe)
        st.write(f"Analysing **{len(tickers)}** tickers. This can take a few minutes.")

        df_res = analyse_universe(tickers, period=period, top_n=top_n)

        if df_res.empty:
            st.error(
                "No results. Try a different universe, ensure your internet connection is ok, "
                "or switch to 2y lookback for more history."
            )
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
