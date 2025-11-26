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

    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ---------- NORMALISATION & INDICATORS ----------

def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure df has simple, non-duplicated OHLC columns.
    Handles occasional MultiIndex or duplicate column names from yfinance.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]

    df = df.loc[:, ~df.columns.duplicated()]

    wanted = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[wanted].copy()

    return df


def _get_ohlc_series(df: pd.DataFrame):
    """
    Return OHLC as Series, even if df had odd structure originally.
    """
    def _col(name: str):
        if name not in df.columns:
            raise KeyError(f"Missing column {name} in price data")
        col = df[name]
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0]
        return col

    open_ = _col("Open")
    high = _col("High")
    low = _col("Low")
    close = _col("Close")
    return open_, high, low, close


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    open_, high, low, close = _get_ohlc_series(df)

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


# ---------- RELAXED SCORING + SL/TP ----------

def score_latest(df: pd.DataFrame):
    # relaxed minimum history
    if df.shape[0] < 130:
        return None

    latest = df.iloc[-1]

    # Price as scalar from normalized Close (last daily price)
    try:
        price = float(df["Close"].iloc[-1])
    except Exception:
        return None

    if np.isnan(price) or price <= 0:
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

    # Direction
    direction = "BUY" if core_signal >= 0 else "SELL"

    # Expected move (heuristic)
    expected_abs_move = float(abs(core_signal) * 5.0)

    # Confidence (0–0.99, not a literal probability)
    confidence_raw = np.tanh(abs(core_signal))
    confidence = float(min(confidence_raw, 0.99))

    # Timeframe by volatility
    vol_20 = df["vol_20"].iloc[-1]
    vol_20 = 0.0 if pd.isna(vol_20) else float(vol_20)
    if atr_pct > 0.03 or vol_20 > 0.025:
        timeframe = "short (1–7 days)"
    else:
        timeframe = "medium (1–4 weeks)"

    # --- Stop loss / Take profit using ATR multiples ---
    sl_atr_mult = 1.5
    tp_atr_mult = 1.5 + 1.5 * confidence  # 1.5x–3x ATR depending on confidence

    sl_pct = atr_pct * sl_atr_mult * 100.0
    tp_pct = atr_pct * tp_atr_mult * 100.0

    if direction == "BUY":
        stop_loss_price = price * (1.0 - sl_pct / 100.0)
        take_profit_price = price * (1.0 + tp_pct / 100.0)
    else:  # SELL
        stop_loss_price = price * (1.0 + sl_pct / 100.0)
        take_profit_price = price * (1.0 - tp_pct / 100.0)

    return {
        "price": float(price),
        "expected_movement_pct": round(expected_abs_move, 2),
        "direction": direction,
        "timeframe": timeframe,
        "confidence": round(confidence, 2),
        "stop_loss_pct": round(sl_pct, 2),
        "take_profit_pct": round(tp_pct, 2),
        "stop_loss_price": round(stop_loss_price, 4),
        "take_profit_price": round(take_profit_price, 4),
    }


# ---------- LIVE PRICE HELPER (SINGLE TICKER) ----------

def get_live_price(ticker: str):
    """
    Best-effort near-live price for a single ticker.
    Tries 1d/1m history first, then fast_info as fallback.
    Returns float or None.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])

        info = getattr(t, "fast_info", None)
        if info is not None:
            last_price = getattr(info, "last_price", None) or getattr(info, "last", None)
            if last_price is not None:
                return float(last_price)

        return None
    except Exception:
        return None


# ---------- SCAN UNIVERSE (PER-TICKER) ----------


@st.cache_data(show_spinner=True)
def analyse_universe(tickers, period="1y"):
    """
    Analyse ALL tickers and return a DataFrame of scored signals.
    No trimming here; filtering and top_n are handled in main().
    """
    results = []
    tickers = list(dict.fromkeys(tickers))

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

        df = _normalize_price_df(df)
        if "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
            continue

        data_ok += 1

        try:
            df = compute_indicators(df)
        except Exception as e:
            st.write(f"{t}: indicator error {e}")
            continue

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
                "stop_loss_pct": scored["stop_loss_pct"],
                "take_profit_pct": scored["take_profit_pct"],
                "stop_loss_price": scored["stop_loss_price"],
                "take_profit_price": scored["take_profit_price"],
            }
        )

    st.write(
        f"Tickers with price data: {data_ok} | "
        f"Tickers with valid score: {scored_ok}"
    )

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    # Sort by TP % but do NOT trim; main() will apply top_n
    df_res = df_res.sort_values(by="take_profit_pct", ascending=False)
    return df_res


# ---------- STREAMLIT UI ----------


def main():
    st.title("UK Top Movers Scanner GB")
    st.markdown(
        "Quant-style heuristic scanner for UK stocks (FTSE 100 / 250 / 350 / AIM-ish). "
        "Uses volatility-scaled momentum, breakouts, and trend filters to highlight "
        "candidates likely to move. **Not financial advice.**"
    )

    # Session-state storage for latest scan
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None
        st.session_state["scan_meta"] = {}

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

    direction_filter = st.selectbox(
        "Show signals",
        ["BUY & SELL", "BUY only", "SELL only"],
        index=0,
    )

    run_scan = st.button("Run scan")

    if run_scan:
        with st.spinner("Fetching tickers…"):
            tickers = get_universe(universe)
        st.write(f"Analysing **{len(tickers)}** tickers.")
        df_all = analyse_universe(tickers, period=period)
        st.session_state["scan_results"] = df_all
        st.session_state["scan_meta"] = {"universe": universe, "period": period}

    # Always read from session_state so results persist across reruns
    df_all = st.session_state.get("scan_results")

    if df_all is None or df_all.empty:
        st.info("Run a scan to see results.")
        return

    # Filter & slice for display
    df_res = df_all.copy()

    if direction_filter == "BUY only":
        df_res = df_res[df_res["direction"] == "BUY"]
    elif direction_filter == "SELL only":
        df_res = df_res[df_res["direction"] == "SELL"]

    if df_res.empty:
        st.warning("No signals match that direction filter.")
        return

    df_res = df_res.sort_values(by="take_profit_pct", ascending=False).head(top_n)

    # Prepare a nicer display version with shorter headers
    df_show = df_res.rename(
        columns={
            "close": "Last daily price",
            "expected_movement_pct": "Exp move %",
            "stop_loss_pct": "SL %",
            "take_profit_pct": "TP %",
            "stop_loss_price": "SL price",
            "take_profit_price": "TP price",
        }
    )

    st.subheader("Top predicted movers")
    st.dataframe(
        df_show.style.format(
            {
                "Last daily price": "{:.2f}",
                "Exp move %": "{:.2f}",
                "confidence": "{:.2f}",
                "SL %": "{:.2f}",
                "TP %": "{:.2f}",
                "SL price": "{:.4f}",
                "TP price": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ----- Single-ticker live refresh -----
    st.markdown("### Live price & SL/TP update for a single ticker")

    tickers_available = df_res["ticker"].tolist()
    selected_ticker = st.selectbox(
        "Select a ticker from the current results",
        tickers_available,
        key="live_ticker_select",
    )

    if st.button("Refresh live price for selected ticker", key="live_refresh_button"):
        live_price = get_live_price(selected_ticker)
        if live_price is None:
            st.warning(
                f"Could not fetch a live price for {selected_ticker}. "
                "Yahoo Finance may not have intraday data or is rate-limiting."
            )
        else:
            row = df_res[df_res["ticker"] == selected_ticker].iloc[0]
            direction = row["direction"]
            sl_pct = row["stop_loss_pct"]
            tp_pct = row["take_profit_pct"]
            last_daily_price = row["close"]

            if direction == "BUY":
                new_sl_price = live_price * (1.0 - sl_pct / 100.0)
                new_tp_price = live_price * (1.0 + tp_pct / 100.0)
            else:  # SELL
                new_sl_price = live_price * (1.0 + sl_pct / 100.0)
                new_tp_price = live_price * (1.0 - tp_pct / 100.0)

            comparison = pd.DataFrame(
                [
                    {
                        "ticker": selected_ticker,
                        "direction": direction,
                        "Last daily price": round(last_daily_price, 4),
                        "Live price": round(live_price, 4),
                        "SL %": round(sl_pct, 2),
                        "TP %": round(tp_pct, 2),
                        "Original SL price": round(row["stop_loss_price"], 4),
                        "Original TP price": round(row["take_profit_price"], 4),
                        "Updated SL price": round(new_sl_price, 4),
                        "Updated TP price": round(new_tp_price, 4),
                    }
                ]
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

    # CSV download uses full column names (df_res)
    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        csv,
        "uk_top_movers_results.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
