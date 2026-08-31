from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="六大技術指標分析台",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* 色彩交由 Streamlit 主題控制，以下只保留不依賴明暗模式的結構樣式。 */
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, .24); }
    h1, h2, h3 { letter-spacing: .01em; }
    p, label, [data-testid="stMetricLabel"] { opacity: .82; }
    .signal-card {
        height: 100%; padding: 1rem 1.1rem;
        border: 1px solid rgba(128, 128, 128, .28);
        border-radius: 12px; background: rgba(128, 128, 128, .07);
    }
    .signal-title { font-size: .82rem; margin-bottom: .35rem; opacity: .72; }
    .signal-value { font-size: 1.15rem; font-weight: 700; margin-bottom: .35rem; }
    .signal-note { font-size: .88rem; line-height: 1.45; opacity: .86; }
    .bull { color: #ff6b7d; }
    .bear { color: #42d7a5; }
    .neutral { color: #e6d58a; opacity: .82; }
    /* 合計分數依正、負、零顯示深紅、深綠、深黃色。 */
    .score-style-marker { display: none; }
    .stElementContainer:has(.score-style-marker) { display: none; }
    [data-testid="stColumn"]:has(.score-positive) [data-testid="stMetricValue"] {
        color: #991b1b;
    }
    [data-testid="stColumn"]:has(.score-negative) [data-testid="stMetricValue"] {
        color: #166534;
    }
    [data-testid="stColumn"]:has(.score-zero) [data-testid="stMetricValue"] {
        color: #a16207;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def download_stock(ticker: str, start: date, end: date) -> pd.DataFrame:
    """下載股價；Yahoo Finance 的 end 為不包含，因此多加一天。"""
    frame = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if isinstance(frame.columns, pd.MultiIndex):
        # 單一股票下載在新版 yfinance 仍可能回傳 MultiIndex。
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if frame.empty or any(column not in frame.columns for column in required):
        return pd.DataFrame()

    # 清洗非交易日或 Yahoo Finance 偶爾回傳的異常 OHLC 列。
    ohlc_cols = ["Open", "High", "Low", "Close"]
    frame = frame[required].dropna(subset=ohlc_cols)
    frame = frame[(frame[ohlc_cols] > 0).all(axis=1)]
    return frame.copy()


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(average_loss.ne(0), 100)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for period in (5, 10, 20):
        data[f"MA{period}"] = data["Close"].rolling(period).mean()

    data["STD20"] = data["Close"].rolling(20).std()
    data["Upper"] = data["MA20"] + data["STD20"] * 2
    data["Lower"] = data["MA20"] - data["STD20"] * 2

    direction = np.sign(data["Close"].diff()).fillna(0)
    data["OBV"] = (direction * data["Volume"]).cumsum()
    data["OBV_MA5"] = data["OBV"].rolling(5).mean()

    low_9 = data["Low"].rolling(9).min()
    high_9 = data["High"].rolling(9).max()
    spread = (high_9 - low_9).replace(0, np.nan)
    data["RSV"] = (data["Close"] - low_9) / spread * 100
    data["K"] = data["RSV"].ewm(alpha=1 / 3, adjust=False).mean()
    data["D"] = data["K"].ewm(alpha=1 / 3, adjust=False).mean()
    data["J"] = 3 * data["K"] - 2 * data["D"]

    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["DIF"] = data["EMA12"] - data["EMA26"]
    data["MACD"] = data["DIF"].ewm(span=9, adjust=False).mean()
    data["MACD_Histogram"] = data["DIF"] - data["MACD"]

    data["RSI5"] = calculate_rsi(data["Close"], 5)
    data["RSI10"] = calculate_rsi(data["Close"], 10)
    data["BIAS10"] = (data["Close"] - data["MA10"]) / data["MA10"] * 100
    data["BIAS20"] = (data["Close"] - data["MA20"]) / data["MA20"] * 100
    data["B10-B20"] = data["BIAS10"] - data["BIAS20"]
    return data


def build_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    # 成交量沿用 Yahoo 台股樣式：漲紅、跌綠、平盤灰。
    conditions = [
        data["Close"] > data["Close"].shift(1),
        data["Close"] < data["Close"].shift(1),
    ]
    choices = ["red", "green"]
    volume_colors = np.select(conditions, choices, default="gray")
    macd_colors = np.where(data["MACD_Histogram"] >= 0, "#ef6475", "#37c994")
    bias_colors = np.where(data["B10-B20"] >= 0, "#ef6475", "#37c994")

    # 排除週末與資料期間內的平日休市日期，讓 K 線時間軸保持連續。
    all_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
    trading_days = data.index
    smholiday = all_days.difference(trading_days)
    smholiday = [d.strftime("%Y-%m-%d") for d in smholiday if d.weekday() < 5]

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.42, 0.13, 0.11, 0.12, 0.11, 0.11],
        specs=[
            [{}],
            [{"secondary_y": True}],
            [{}],
            [{"secondary_y": True}],
            [{}],
            [{}],
        ],
        subplot_titles=("價格・均線・布林通道", "成交量・OBV", "KDJ", "MACD", "RSI", "BIAS"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="K 線", increasing_line_color="#ef6475", decreasing_line_color="#37c994",
        ), row=1, col=1,
    )
    for column, color in (("MA5", "#36c5f0"), ("MA10", "#b48df2"), ("MA20", "#f4c95d")):
        fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=dict(color=color, width=1.2)), row=1, col=1)
    for column, dash in (("Upper", "dot"), ("Lower", "dot")):
        fig.add_trace(
            go.Scatter(x=data.index, y=data[column], name=column, line=dict(color="#7184a8", width=1, dash=dash)),
            row=1, col=1,
        )

    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="成交量", marker_color=volume_colors, opacity=.65), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["OBV"], name="OBV", line=dict(color="#36c5f0", width=1.3)), row=2, col=1, secondary_y=True)

    for column, color in (("K", "#36c5f0"), ("D", "#b48df2"), ("J", "#f4c95d")):
        fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=dict(color=color, width=1.1)), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#50617f", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#50617f", row=3, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["DIF"], name="DIF", line=dict(color="#36c5f0", width=1.1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#b48df2", width=1.1)), row=4, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["MACD_Histogram"], name="MACD 柱", marker_color=macd_colors, opacity=.65), row=4, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="#50617f", line_width=1, row=4, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["RSI5"], name="RSI5", line=dict(color="#36c5f0", width=1.1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI10"], name="RSI10", line=dict(color="#b48df2", width=1.1)), row=5, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#50617f", row=5, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#50617f", row=5, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["BIAS10"], name="BIAS10", line=dict(color="#36c5f0", width=1.1)), row=6, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["BIAS20"], name="BIAS20", line=dict(color="#b48df2", width=1.1)), row=6, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["B10-B20"], name="B10-B20", marker_color=bias_colors, opacity=.55), row=6, col=1)
    fig.add_hline(y=0, line_color="#50617f", line_width=1, row=6, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker}｜六大技術指標", x=.01),
        height=1200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft JhengHei, Noto Sans TC, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1,
            yanchor="top",
        ),
        margin=dict(l=42, r=145, t=92, b=38),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=smholiday)],
        showgrid=True, gridcolor="rgba(128,128,128,.18)", zeroline=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.025,
            bgcolor="rgba(128,128,128,.08)",
            bordercolor="rgba(128,128,128,.28)",
            borderwidth=1,
        ),
        row=6,
        col=1,
    )
    return fig


def signal(label: str, score: int, detail: str) -> dict[str, object]:
    state = "偏多" if score > 0 else "偏空" if score < 0 else "中性"
    return {"指標": label, "判讀": state, "參考說明": detail, "分數": score}


def analyze_indicators(data: pd.DataFrame) -> list[dict[str, object]]:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    results: list[dict[str, object]] = []

    if pd.isna(latest["MA20"]):
        results.append(signal("均線＋布林通道", 0, "資料不足 20 個交易日，暫無法判讀。"))
    elif latest["Close"] > latest["MA20"] and latest["MA5"] > latest["MA10"]:
        note = "收盤站上 MA20，且短均線高於中期均線；趨勢動能較正向。"
        if latest["Close"] > latest["Upper"]:
            note += " 已突破布林上軌，留意短線過熱與拉回。"
        results.append(signal("均線＋布林通道", 1, note))
    elif latest["Close"] < latest["MA20"] and latest["MA5"] < latest["MA10"]:
        note = "收盤位於 MA20 下方，且短均線低於中期均線；趨勢相對弱勢。"
        if latest["Close"] < latest["Lower"]:
            note += " 已跌破布林下軌，留意超跌反彈與風險。"
        results.append(signal("均線＋布林通道", -1, note))
    else:
        results.append(signal("均線＋布林通道", 0, "價格與均線排列不一致，目前偏向盤整。"))

    obv_change = latest["OBV"] - data["OBV"].iloc[-6] if len(data) >= 6 else latest["OBV"] - data["OBV"].iloc[0]
    price_change = latest["Close"] - data["Close"].iloc[-6] if len(data) >= 6 else latest["Close"] - data["Close"].iloc[0]
    if obv_change > 0 and price_change >= 0:
        results.append(signal("成交量＋OBV", 1, "近 5 日價格與 OBV 同步走高，量價結構偏正向。"))
    elif obv_change < 0 and price_change <= 0:
        results.append(signal("成交量＋OBV", -1, "近 5 日價格與 OBV 同步走低，賣壓相對明顯。"))
    else:
        results.append(signal("成交量＋OBV", 0, "價格與 OBV 走勢分歧，宜等待量價重新同步。"))

    if latest["K"] > latest["D"] and previous["K"] <= previous["D"]:
        results.append(signal("KDJ", 1, f"K 值向上穿越 D 值（金叉）；目前 K={latest['K']:.1f}、D={latest['D']:.1f}。"))
    elif latest["K"] < latest["D"] and previous["K"] >= previous["D"]:
        results.append(signal("KDJ", -1, f"K 值向下穿越 D 值（死叉）；目前 K={latest['K']:.1f}、D={latest['D']:.1f}。"))
    elif latest["K"] >= 80:
        results.append(signal("KDJ", 0, f"K={latest['K']:.1f}，位於高檔區，留意動能鈍化或回檔。"))
    elif latest["K"] <= 20:
        results.append(signal("KDJ", 0, f"K={latest['K']:.1f}，位於低檔區，留意止跌訊號。"))
    else:
        results.append(signal("KDJ", 1 if latest["K"] > latest["D"] else -1, f"K={latest['K']:.1f}、D={latest['D']:.1f}，依 K/D 相對位置判讀。"))

    macd_score = 1 if latest["DIF"] > latest["MACD"] and latest["MACD_Histogram"] > 0 else -1 if latest["DIF"] < latest["MACD"] and latest["MACD_Histogram"] < 0 else 0
    results.append(signal("MACD", macd_score, f"DIF={latest['DIF']:.2f}、訊號線={latest['MACD']:.2f}、柱狀體={latest['MACD_Histogram']:.2f}。"))

    rsi = latest["RSI10"]
    if rsi >= 70:
        results.append(signal("RSI", 0, f"RSI10={rsi:.1f}，進入常見超買區，強勢但須注意追高風險。"))
    elif rsi <= 30:
        results.append(signal("RSI", 0, f"RSI10={rsi:.1f}，進入常見超賣區，弱勢但可能醞釀反彈。"))
    else:
        results.append(signal("RSI", 1 if rsi >= 50 else -1, f"RSI10={rsi:.1f}，位於 50 {'以上' if rsi >= 50 else '以下'}。"))

    bias_score = 1 if latest["BIAS10"] > latest["BIAS20"] and latest["B10-B20"] > 0 else -1 if latest["BIAS10"] < latest["BIAS20"] and latest["B10-B20"] < 0 else 0
    results.append(signal("BIAS", bias_score, f"BIAS10={latest['BIAS10']:.2f}%、BIAS20={latest['BIAS20']:.2f}%；乖離愈大代表價格偏離均線愈遠。"))
    return results


today = date.today()
default_start = today - timedelta(days=365)

with st.sidebar:
    st.markdown("## 查詢條件")
    with st.form("stock_query"):
        ticker_input = st.text_input("股票代號", value="2330.TW", help="例如：2330.TW、AAPL、TSLA").strip().upper()
        start_input = st.date_input("開始日期", value=default_start, max_value=today)
        end_input = st.date_input("結束日期", value=today, max_value=today)
        submitted = st.form_submit_button("開始分析", width="stretch")
    st.caption("台股請在代號後加 .TW；上櫃股票通常使用 .TWO。")

if submitted or "query" not in st.session_state:
    st.session_state.query = (ticker_input or "2330.TW", start_input, end_input)

ticker, start_date, end_date = st.session_state.query

st.title("六大技術指標分析台")
st.caption("以同一張時間軸檢視趨勢、量價與動能；預設查詢近一年資料。")

if start_date >= end_date:
    st.error("開始日期必須早於結束日期，請調整後再按「開始分析」。")
    st.stop()

with st.spinner(f"正在取得 {ticker} 的市場資料…"):
    raw_data = download_stock(ticker, start_date, end_date)

if raw_data.empty:
    st.error("查無股價資料。請確認股票代號、交易所後綴與日期範圍是否正確。")
    st.stop()

data = add_indicators(raw_data)
latest = data.iloc[-1]
previous_close = data["Close"].iloc[-2] if len(data) >= 2 else latest["Close"]
change = latest["Close"] - previous_close
change_pct = change / previous_close * 100 if previous_close else 0

metric_columns = st.columns(4)
metric_columns[0].metric("最新收盤", f"{latest['Close']:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
metric_columns[1].metric("區間最高", f"{data['High'].max():,.2f}")
metric_columns[2].metric("區間最低", f"{data['Low'].min():,.2f}")
metric_columns[3].metric("最新成交量", f"{latest['Volume']:,.0f}")

st.plotly_chart(
    build_chart(data, ticker),
    width="stretch",
    theme="streamlit",
    config={"displaylogo": False},
)

st.markdown("## 🚦 六大指標分析參考")
results = analyze_indicators(data)
indicator_icons = {
    "均線＋布林通道": "📈",
    "成交量＋OBV": "📊",
    "KDJ": "🎯",
    "MACD": "〽️",
    "RSI": "⚡",
    "BIAS": "↔️",
}
cards = st.columns(3)
for index, result in enumerate(results):
    css_class = "bull" if result["分數"] > 0 else "bear" if result["分數"] < 0 else "neutral"
    status_icon = "🔴" if result["分數"] > 0 else "🟢" if result["分數"] < 0 else "🟡"
    indicator_icon = indicator_icons.get(result["指標"], "📌")
    with cards[index % 3]:
        st.markdown(
            f"""
            <div class="signal-card">
              <div class="signal-title">{indicator_icon} {result['指標']}</div>
              <div class="signal-value {css_class}">{status_icon} {result['判讀']}</div>
              <div class="signal-note">{result['參考說明']}</div>
            </div><br>
            """,
            unsafe_allow_html=True,
        )

total_score = sum(int(item["分數"]) for item in results)
bull_count = sum(1 for item in results if item["分數"] > 0)
bear_count = sum(1 for item in results if item["分數"] < 0)
neutral_count = sum(1 for item in results if item["分數"] == 0)
overall = "整體偏多" if total_score >= 2 else "整體偏空" if total_score <= -2 else "多空訊號混合"
overall_icon = "🔴" if total_score >= 2 else "🟢" if total_score <= -2 else "🟡"
summary_column, score_column = st.columns([2, 1])
with summary_column:
    st.info(
        f"綜合參考：{overall_icon} **{overall}** ｜ "
        f"🔴 偏多 **{bull_count}** 項 ｜ "
        f"🟢 偏空 **{bear_count}** 項 ｜ "
        f"🟡 中性 **{neutral_count}** 項 ｜ "
        f"共 **{len(results)}** 項"
    )
with score_column:
    score_color_class = (
        "score-positive" if total_score > 0
        else "score-negative" if total_score < 0
        else "score-zero"
    )
    st.markdown(
        f'<span class="score-style-marker {score_color_class}"></span>',
        unsafe_allow_html=True,
    )
    st.metric(
        "🧮 合計分數",
        f"{total_score:+d} 分",
        help="單項偏多 +1、偏空 -1、中性 0。",
    )

with st.expander("查看判讀表與最近資料"):
    result_frame = pd.DataFrame(results).drop(columns="分數")
    st.dataframe(result_frame, width="stretch", hide_index=True)
    st.dataframe(data.tail(20).sort_index(ascending=False), width="stretch")

st.warning("以上內容依歷史價格與技術指標自動計算，僅供研究與教學參考，不構成投資建議。")
