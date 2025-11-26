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
    constituents table that has a 'Ticker' / 'EPIC' / 'Symbol' / 'Code' column.
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
        # filter out junk: short-ish, uppercase-ish, no spaces
        series = series[series.str.len().between(1, 6)]
        series = series[~series.str.contains(" ", regex=False)]
        tickers = [
            t
            for t in series.tolist()
            if t and t.lower() not in ("ticker", "epic", "symbol", "code")
        ]
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
            tickers = [
                t
                for t in series.tolist()
                if t and t.lower() not in ("epic", "ticker", "symbol", "code")
            ]
            if len(tickers) >= 40:
                return [t + ".L" for t in tickers]
    except Exception:
        pass

    return []


def _get_aim_from_yahoo_screener() -> list:
    """
    Fallback: use Yahoo Finance screener API to get UK/LSE small caps.
    Not strictly 'AIM 100', but gives an AIM-like universe.
    Fully wrapped in try/except; returns [] on failure.
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
                    # proxy for smaller caps
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
        quotes = (
            js.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
        symbols = [q.get("symbol") for q in quotes if isinstance(q, dict) and q.get("symbol")]
        symbols = [s for s in symbols if s.endswith(".L")]
        return symbols
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_aim100_tickers() -> list:
    """
    AIM 100 / AIM-like universe:
    1) Try DigitalLook.
    2) If that fails or is too small, fallback to Yahoo screener.
    3) If both fail, return [] and show a warning.
    """
    tickers = _get_aim_from_digitallook()
    if len(tickers) >= 40:
        return tickers

    yf_tickers = _get_aim_from_yahoo_screener()
    if len(yf_tickers) >= 40:
        return yf_tickers

    st.warning(
        "AIM 100 / AIM-like tickers could not be reliably fetched. "
        "AIM 100 and All UK will currently only include FTSE names."
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


# ---------- INDICATORS & RELAXED SCORING ----------


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    returns = close.pct_change()
    df["ret"] = returns
    df["vol_20"] = returns.rolling(20, min_periods=10).std()
    df["vol_60"] = returns.rolling(60, min_periods=20).std()

    df["mom_20"] = close.pct_change(20)
    df["mom_60"] = close.pct_change(60)
    df["ts_mom_20"] = df["mom_20"] / (df["vol_20"] * np.sqrt(20))
    df["ts_mom_60"] = df["mom_60"] / (df["vol_60"] * np.sqrt(60))

    high55 = close.rolling(55, min_periods=20).max()
    low55 = close.rolling(55, min_periods=20).min()
    df["breakout_up"] = (close - high55) / high55
    df["breakout_down"] = (close - low55) / low55

    df["ma_50"] = close.rolling(50, min_periods=25).mean()
    df["ma_200"] = close.rolling(200, min_periods=60).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=7).mean()
    df["atr_pct"] = df["ATR"] / close

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def score_latest(df: pd.DataFrame):
    # relaxed minimum history (about 6-8 months)
    if df.shape[0] < 130:
        return None

    latest = df.iloc[-1]

    # Core fields
    price = latest.get("Close", np.nan)
    if pd.isna(price) or price <= 0:
        return None

    ts_mom_20 = latest.get("ts_mom_20", 0.0)
    ts_mom_60 = latest.get("ts_mom_60", 0.0)
    breakout_up = latest.get("breakout_up", 0.0)
    breakout_down = latest.get("breakout_down", 0.0)
    ma50 = latest.get("ma_50", np.nan)
    ma200 = latest.get("ma_200", np.nan)
    atr_pct = latest.get("atr_pct", np.nan)
    rsi = latest.get("RSI", np.nan)

    ts_mom_20 = 0.0 if pd.isna(ts_mom_20) else float(ts_mom_20)
    ts_mom_60 = 0.0 if pd.isna(ts_mom_60) else float(ts_mom_60)
    breakout_up = 0.0 if pd.isna(breakout_up) else float(breakout_up)
    breakout_down = 0.0 if pd.isna(breakout_down) else float(breakout_down)

    # Trend regime
    trend_regime = 0
    if not pd.isna(ma50) and not pd.isna(ma200):
        ma50 = float(ma50)
        ma200 = float(ma200)
        if price > ma50 > ma200:
            trend_regime = 1
        elif price < ma50 < ma200:
            trend_regime = -1

    # ATR pct fallback
    if pd.isna(atr_pct) or atr_pct <= 0:
        fallback_vol = df["ret"].rolling(20).std().iloc[-1]
        atr_pct = float(fallback_vol) if not pd.isna(fallback_vol) else 0.02
    else:
        atr_pct = float(atr_pct)

    # RSI fallback
    if pd.isna(rsi):
        rsi = 50.0
    else:
        rsi = float(rsi)

    # Core signal
    core_signal = (
        0.6 * ts_mom_20 +
        0.4 * ts_mom_60 +
        1.0 * breakout_up -
        1.0 * breakout_down
    )

    # Trend weighting
    if trend_regime == 1 and core_signal > 0:
        core_signal *= 1.3
    elif trend_regime == -1 and core_signal < 0:
        core_signal *= 1.3
    elif trend_regime == 0:
        core_signal *= 0.9

    # Very extreme RSI only
    if rsi > 85 or rsi < 15:
        core_signal *= 0.85

    # Direction & expected move (heuristic)
    direction = "BUY" if core_signal >= 0 else "SELL"
    expected_abs_move = float(abs(core_signal) * 5.0)

    # Timeframe by volatility
    vol_20 = df["vol_20"].iloc[-1]
    vol_20 = 0.0 if pd.isna(vol_20) else float(vol_20)
    if atr_pct > 0.03 or vol_20 > 0.025:
        timeframe = "short (1–7 days)"
    else:
        timeframe = "medium (1–4 weeks)"

    confidence = float(np.tanh(abs(core_signal)))

    return {
        "price": float(price),
        "expected_movement_pct": round(expected_abs_move, 2),
        "direction": direction,
        "timeframe": timeframe,
        "confidence": round(confidence, 2),
    }


# ---------- SCAN UNIVERSE (SIMPLE, PER-TICKER) ----------


@st.cache_data(show_spinner=True)
def analyse_universe(tickers, period="1y", top_n=10):
    results = []
    tickers = list(dict.fromkeys(tickers))  # de-duplicate

    n = len(tickers)
    if n == 0:
        return pd.DataFrame()

    progress = st.progress(0)
    data_ok = 0
    scored_ok = 0

    for idx, t in enumerate(tickers, start=1):
        progress.progress(idx / n)
        try:
            df = yf.download(
                t,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            st.write(f"{t}: download error {e}")
            continue

        if df is None or df.empty:
            continue

        data_ok += 1

        df = compute_indicators(df)
        scored = score_latest(df)
        if not scored:
            continue

        scored_ok += 1

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

    st.write(
        f"Tickers with price data: {data_ok} | "
        f"Tickers with valid score: {scored_ok}"
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
        "Quant-style heuristic scanner for UK stocks (FTSE 100 / 250 / 350 / AIM-ish). "
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
        help="Longer lookback gives better trend information but takes longer.",
    )

    if st.button("Run scan"):
        with st.spinner("Fetching tickers…"):
            tickers = get_universe(universe)
        st.write(f"Analysing **{len(tickers)}** tickers.")

        df_res = analyse_universe(tickers, period=period, top_n=top_n)

        if df_res.empty:
            st.error(
                "No results. Check the counts just above: "
                "if 'tickers with price data' is 0, Yahoo Finance may be blocking or failing. "
                "If data > 0 but 'valid score' is 0, we can relax filters further."
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
