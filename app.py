
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

# ============================================================
# 一、UI 主题：蓝金配色 + 卡片样式（基础样式，暗黑模式下局部覆盖）
# ============================================================

BLUE = "#0055A4"
GOLD = "#CFAF70"
BG_LIGHT = "#F5F7FA"

BASE_CSS = f"""
<style>
body {{
    background-color: {BG_LIGHT};
}}
h1 {{
    color: {BLUE} !important;
    font-weight: 900 !important;
}}
h2, h3, h4 {{
    color: {BLUE} !important;
}}
.big-number {{
    font-size: 32px;
    font-weight: 700;
    color: {BLUE};
    text-align: center;
}}
.big-number-gold {{
    font-size: 32px;
    font-weight: 700;
    color: {GOLD};
    text-align: center;
}}
.card {{
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #E0E4EA;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}
.card-title {{
    font-size: 14px;
    color: #666;
    text-align: center;
    margin-top: 8px;
}}
.card-sub {{
    font-size: 12px;
    color: #999;
    text-align: center;
}}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ============================================================
# 二、列名映射（英文 → 中文）
# ============================================================

COLUMN_NAME_MAP = {
    "date": "日期",
    "cash_in": "现金流入",
    "cash_out": "现金流出",
    "net_cash_flow": "净现金流",
    "sales": "销售收入",
    "project_spend": "项目支出",
    "tax_payment": "税费缴纳",
}

FEATURE_NAME_MAP = COLUMN_NAME_MAP


def format_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def styled_table(df: pd.DataFrame):
    if df.empty:
        return df.style

    df2 = df.copy()
    if "日期" in df2.columns:
        df2["日期"] = format_date_series(df2["日期"])

    numeric_cols = df2.select_dtypes(include=[np.number]).columns

    styler = df2.style
    if len(numeric_cols) > 0:
        styler = styler.format("{:.2f}", subset=numeric_cols)
        styler = styler.set_properties(**{"text-align": "center"}, subset=numeric_cols)

    return styler


# ============================================================
# 三、数据生成 & 预处理
# ============================================================

def generate_synthetic_data(n_days: int = 730) -> pd.DataFrame:
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    t = np.arange(n_days)

    sales = (
        200000 + 500 * t +
        50000 * np.sin(2 * np.pi * t / 365) +
        20000 * np.random.randn(n_days)
    )

    project_spend = (
        80000 + 10000 * np.sin(2 * np.pi * t / 180) +
        15000 * np.random.randn(n_days)
    )

    spikes = np.random.choice(n_days, size=15, replace=False)
    project_spend[spikes] += np.random.uniform(50000, 200000, len(spikes))

    tax_payment = np.zeros(n_days)
    for i, d in enumerate(dates):
        if d.day == 15:
            tax_payment[i] = 50000 + 20000 * np.random.rand()

    cash_in = sales * np.random.uniform(0.7, 0.9) + np.random.randn(n_days) * 20000
    cash_out = project_spend + tax_payment + np.random.uniform(0.4, 0.6) * 0.5 * sales

    df = pd.DataFrame({
        "date": dates,
        "cash_in": cash_in,
        "cash_out": cash_out,
        "sales": sales,
        "project_spend": project_spend,
        "tax_payment": tax_payment
    })
    df["net_cash_flow"] = df["cash_in"] - df["cash_out"]

    return df


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df = df.set_index("date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df = df.reset_index()

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate().ffill().bfill()

    return df


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)

    if "date" not in df.columns:
        raise ValueError("CSV 必须包含 date 列")

    if "net_cash_flow" not in df.columns:
        if "cash_in" in df.columns and "cash_out" in df.columns:
            df["net_cash_flow"] = df["cash_in"] - df["cash_out"]
        else:
            raise ValueError("缺少 net_cash_flow 且无法自动生成")

    return basic_preprocess(df).dropna().reset_index(drop=True)


# ============================================================
# 四、LSTM 相关函数
# ============================================================

def create_sequences(X, y, window_size=60):
    xs, ys = [], []
    for i in range(len(X) - window_size):
        xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(xs), np.array(ys)


def build_lstm_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    return model


def train_lstm_model(df, feature_cols, target="net_cash_flow",
                     window=60, epochs=20, batch_size=32):

    feature_cols = list(dict.fromkeys(feature_cols))
    X_raw = df[feature_cols].astype(float).values
    y_raw = df[[target]].astype(float).values.reshape(-1, 1)

    fs = MinMaxScaler()
    ts = MinMaxScaler()

    X_scaled = fs.fit_transform(X_raw)
    y_scaled = ts.fit_transform(y_raw)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window)

    if len(X_seq) < 10:
        raise ValueError("样本量过少，无法训练 LSTM 模型，请提供更多数据。")

    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = build_lstm_model((window, X_seq.shape[2]))

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    pred_scaled = model.predict(X_val, verbose=0)
    y_true = ts.inverse_transform(y_val).reshape(-1)
    y_pred = ts.inverse_transform(pred_scaled).reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    hist_df = df[["date", target]].copy()

    return model, fs, ts, X_scaled, y_scaled, hist_df, {"mae": mae, "rmse": rmse}


# ============================================================
# 五、MC Dropout 多步预测
# ============================================================

def mc_dropout_forecast_batch(model,
                              last_window,
                              scaler,
                              steps=60,
                              n_samples=30):
    means, stds = [], []
    cur = last_window.copy()

    for _ in range(steps):
        batch = np.repeat(cur[np.newaxis], n_samples, axis=0)
        preds = model(batch, training=True).numpy().reshape(-1)

        mu = preds.mean()
        sigma = preds.std(ddof=1) if len(preds) > 1 else 0.0

        means.append(mu)
        stds.append(sigma)

        nxt = cur[-1].copy()
        nxt[0] = mu
        cur = np.vstack([cur[1:], nxt])

    means = np.array(means).reshape(-1, 1)
    stds = np.array(stds).reshape(-1, 1)

    upper = means + 1.96 * stds
    lower = means - 1.96 * stds

    return (
        scaler.inverse_transform(means).reshape(-1),
        scaler.inverse_transform(lower).reshape(-1),
        scaler.inverse_transform(upper).reshape(-1)
    )


# ============================================================
# 六、异常检测 & 敏感性分析
# ============================================================

def detect_anomalies_combined(dates, values, z_thresh=3.0, iqr_k=1.5):
    v = np.asarray(values, float).reshape(-1)
    d = np.asarray(dates)

    mu = float(v.mean())
    sigma = float(v.std(ddof=1))
    z = np.zeros_like(v) if sigma == 0 else (v - mu) / sigma
    z_mask = (np.abs(z) >= z_thresh).reshape(-1)

    Q1, Q3 = np.percentile(v, 25), np.percentile(v, 75)
    IQR = Q3 - Q1
    low, high = Q1 - iqr_k * IQR, Q3 + iqr_k * IQR
    iqr_mask = ((v < low) | (v > high)).reshape(-1)

    mask = (z_mask | iqr_mask).reshape(-1)

    if not mask.any():
        return pd.DataFrame()

    severity = np.abs(z) / max(z_thresh, 1e-6) + iqr_mask.astype(float)

    df = pd.DataFrame({
        "date": d[mask],
        "value": v[mask],
        "zscore": z[mask],
        "iqr_flag": iqr_mask[mask],
        "z_flag": z_mask[mask],
        "severity": severity[mask]
    })

    return df.sort_values("severity", ascending=False).reset_index(drop=True)


def feature_sensitivity_last_window(model,
                                    window,
                                    feature_names,
                                    scaler,
                                    delta=0.1):
    base_scaled = float(model(window[np.newaxis], training=False).numpy().squeeze())
    base = scaler.inverse_transform([[base_scaled]])[0, 0]

    results = []
    for i, name in enumerate(feature_names):
        pert = window.copy()
        pert[:, i] *= (1 + delta)

        new_scaled = float(model(pert[np.newaxis], training=False).numpy().squeeze())
        new_val = scaler.inverse_transform([[new_scaled]])[0, 0]

        results.append({"feature": name, "change": new_val - base})

    return pd.DataFrame(results).sort_values("change", ascending=False)


def anomalies_to_chinese(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.rename(columns={
        "date": "日期",
        "value": "净现金流",
        "zscore": "Z值",
        "iqr_flag": "IQR异常",
        "z_flag": "Z异常",
        "severity": "异常强度"
    }, inplace=True)
    out["日期"] = format_date_series(out["日期"])
    return out


def sensitivity_to_chinese(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["feature"] = out["feature"].apply(lambda x: FEATURE_NAME_MAP.get(x, x))
    out.rename(columns={"feature": "特征", "change": "影响值"}, inplace=True)
    return out


# ============================================================
# 七、Plotly 图表构建（支持多模式）
# ============================================================

def build_forecast_figure(
    history,
    forecast_df,
    scenario_name,
    viz_mode="标准模式",
):
    dates_hist = format_date_series(history["date"])
    dates_fut = forecast_df["日期"]
    hist_values = history["net_cash_flow"]

    if viz_mode == "暗黑模式":
        template = "plotly_dark"
        bg_color = "#0d1117"
        plot_bg = "#0d1117"
        line_hist_color = "#00BFFF"
        line_pred_color = "#8A2BE2"
        band_color = "rgba(0, 191, 255, 0.2)"
        zero_line_color = "#FF4B4B"
    elif viz_mode == "司库驾驶舱模式":
        template = "plotly_white"
        bg_color = "#F2F4F8"
        plot_bg = "white"
        line_hist_color = BLUE
        line_pred_color = GOLD
        band_color = "rgba(0, 85, 164, 0.15)"
        zero_line_color = "red"
    else:  # 标准模式
        template = "plotly_white"
        bg_color = BG_LIGHT
        plot_bg = "white"
        line_hist_color = BLUE
        line_pred_color = "#FF7F0E"
        band_color = "rgba(0, 85, 164, 0.15)"
        zero_line_color = "red"

    fig = go.Figure()

    # 历史净现金流
    fig.add_trace(
        go.Scatter(
            x=dates_hist,
            y=hist_values,
            mode="lines",
            name="历史净现金流",
            line=dict(color=line_hist_color, width=2),
            hovertemplate="日期=%{x}<br>净现金流=%{y:,.2f}<extra></extra>",
        )
    )

    # 预测均值
    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["预测均值"],
            mode="lines",
            name=f"{scenario_name}情景预测均值",
            line=dict(color=line_pred_color, width=2, dash="dash"),
            hovertemplate="日期=%{x}<br>预测净现金流=%{y:,.2f}<extra></extra>",
        )
    )

    # 置信区间带
    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["上界（95%）"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["下界（95%）"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=band_color,
            name="95% 置信区间",
            hovertemplate="日期=%{x}<br>区间=[%{y:,.2f}, %{customdata:,.2f}]<extra></extra>",
            customdata=forecast_df["上界（95%）"],
        )
    )

    # 资金缺口预警红线
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=zero_line_color,
        annotation_text="资金缺口预警红线（0）",
        annotation_position="top left",
        annotation_font=dict(color=zero_line_color),
    )

    fig.update_layout(
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(
            title="日期",
            tickangle=-45,
            showgrid=True,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="净现金流",
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        title=dict(
            text=f"现金流预测（集成模型 + 置信区间 + 情景：{scenario_name}）",
            x=0.5,
            xanchor="center",
        ),
    )

    return fig


# ============================================================
# 八、Streamlit 主程序（支持模式切换）
# ============================================================

def main():
    st.set_page_config(page_title="AI 赋能司库：现金流预测系统", layout="wide")

    # 侧边栏模式切换
    st.sidebar.header("🎛 显示与预测模式")
    viz_mode = st.sidebar.radio(
        "可视化模式",
        ["标准模式", "司库驾驶舱模式", "暗黑模式"],
        index=0,
        help="可在标准 / 司库驾驶舱 / 暗黑大屏三种模式间切换展示效果。",
    )

    # 侧边栏参数
    st.sidebar.header("⚙ 参数设置")

    uploaded_file = st.sidebar.file_uploader("📤 上传现金流 CSV（含 date 列）", type=["csv"])
    use_synthetic = st.sidebar.checkbox(
        "使用系统模拟数据（忽略上传文件）",
        value=(uploaded_file is None)
    )

    window_size = st.sidebar.slider("时间窗口长度（天）", 30, 120, 60, step=5)
    forecast_days = st.sidebar.slider("预测天数", 7, 180, 90, step=7)
    epochs = st.sidebar.slider("训练轮数（Epoch）", 5, 50, 20, step=5)
    n_samples = st.sidebar.slider("Monte-Carlo Dropout 次数", 10, 100, 30, step=10)

    # 情景切换
    scenario = st.sidebar.radio(
        "情景模式",
        ["谨慎", "中性", "乐观"],
        index=1,
        help="谨慎：在预测基础上下调 10%；乐观：在预测基础上上调 10%。"
    )

    run_button = st.sidebar.button("🚀 开始训练与预测")

    # 顶部 LOGO + 标题区
    col_logo, col_title, col_mode = st.columns([1, 4, 2])

    with col_logo:
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=80)
        else:
            st.markdown(
                f"<div style='width:80px;height:80px;border-radius:12px;"
                f"background:{BLUE};display:flex;align-items:center;justify-content:center;'>"
                f"<span style='color:white;font-weight:bold;'>LOGO</span></div>",
                unsafe_allow_html=True
            )

    with col_title:
        st.markdown(
            f"""
            <h1>AI 赋能司库 · 现金流预测系统</h1>
            <h4 style="color:{GOLD};margin-top:-8px;">现代投资财务司库管理案例系统</h4>
            <h5 style="color:#777;margin-top:-12px;">案例：AI赋能司库内部资金管理——基于LSTM的现金流预测系统</h5>
            """,
            unsafe_allow_html=True
        )

    with col_mode:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">当前可视化模式</div>
                <div class="big-number-gold">{viz_mode}</div>
                <div class="card-sub">可在左侧切换展示风格</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 数据加载
    if use_synthetic:
        df = generate_synthetic_data()
    else:
        if uploaded_file is None:
            st.warning("请上传 CSV 文件或勾选使用模拟数据。")
            return
        try:
            df = load_data_from_upload(uploaded_file)
        except Exception as e:
            st.error(f"数据加载失败：{e}")
            return

    df = basic_preprocess(df)

    # =============== 司库驾驶舱大屏模式：顶部多指标布局 ===============
    if viz_mode == "司库驾驶舱模式":
        st.subheader("📊 司库驾驶舱 · 资金全景总览")

        last_net_cf = float(df["net_cash_flow"].iloc[-1])
        last30_std = float(df["net_cash_flow"].tail(30).std())
        avg7 = float(df["net_cash_flow"].tail(7).mean())
        max_in = float(df["cash_in"].tail(30).max())
        max_out = float(df["cash_out"].tail(30).max())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number">{last_net_cf:,.2f}</div>
                    <div class="card-title">今日净现金流（元）</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number">{avg7:,.2f}</div>
                    <div class="card-title">近7日平均净现金流（元）</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number">{last30_std:,.2f}</div>
                    <div class="card-title">近30日净现金流波动率</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number-gold">{max_out:,.2f}</div>
                    <div class="card-title">近30日单日最大现金流出</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        # 标准 / 暗黑模式沿用之前的 Dashboard
        st.subheader("📊 资金监控大屏 Dashboard")

        last_net_cf = float(df["net_cash_flow"].iloc[-1])
        last30_std = float(df["net_cash_flow"].tail(30).std())
        avg7 = float(df["net_cash_flow"].tail(7).mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number">{last_net_cf:,.2f}</div>
                    <div class="card-title">今日净现金流（元）</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number">{last30_std:,.2f}</div>
                    <div class="card-title">近30日净现金流波动率</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="big-number-gold">{avg7:,.2f}</div>
                    <div class="card-title">近7日平均净现金流（元）</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # AI 观点
    st.markdown("#### 💡 AI 司库观点")
    if avg7 < 0:
        st.error("未来短期净现金流偏弱，建议提前统筹资金调度、压降支出并加快回款。")
    else:
        st.success("未来短期净现金流整体平稳偏正，资金安全边际较为充足，可稳步推进既定经营计划。")

    # 数据预览
    st.subheader("📁 数据预览（中文表头）")
    preview = df[["date", "cash_in", "cash_out", "net_cash_flow"]].copy()
    preview.rename(columns=COLUMN_NAME_MAP, inplace=True)
    preview["日期"] = format_date_series(preview["日期"])
    st.dataframe(styled_table(preview), use_container_width=True)

    if not run_button:
        st.info("请在左侧设置参数后，点击“开始训练与预测”。")
        return

    # 模型训练
    target = "net_cash_flow"
    multi_features = [
        "net_cash_flow", "sales", "project_spend",
        "tax_payment", "cash_in", "cash_out"
    ]
    multi_features = [c for c in multi_features if c in df.columns]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔹 单变量 LSTM（仅净现金流）")
        with st.spinner("正在训练单变量模型…"):
            m1, fs1, ts1, X1, y1, hist1, eval1 = train_lstm_model(
                df, [target], target, window_size, epochs
            )
        st.write(f"MAE：{eval1['mae']:.2f}")
        st.write(f"RMSE：{eval1['rmse']:.2f}")
    with col2:
        st.markdown("### 🔸 多特征 LSTM（净现金流 + 业务特征）")
        with st.spinner("正在训练多特征模型…"):
            m2, fs2, ts2, X2, y2, hist2, eval2 = train_lstm_model(
                df, multi_features, target, window_size, epochs
            )
        st.write(f"MAE：{eval2['mae']:.2f}")
        st.write(f"RMSE：{eval2['rmse']:.2f}")

    history = hist2.copy()

    # 多步预测 + 集成 + 情景
    st.subheader("🔮 现金流预测（集成模型 + 置信区间 + 情景）")
    with st.spinner("正在进行多步预测与不确定性估计…"):
        last1 = X1[-window_size:]
        last2 = X2[-window_size:]

        mean1, low1, high1 = mc_dropout_forecast_batch(
            m1, last1, ts1, forecast_days, n_samples
        )
        mean2, low2, high2 = mc_dropout_forecast_batch(
            m2, last2, ts2, forecast_days, n_samples
        )

        inv1 = 1 / (eval1["rmse"] + 1e-6)
        inv2 = 1 / (eval2["rmse"] + 1e-6)
        w1 = inv1 / (inv1 + inv2)
        w2 = inv2 / (inv2 + inv2) if (inv2 + inv2) != 0 else 0.5  # 防御性写法（不太可能为0）

        last_date = history["date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

        base_mean = w1 * mean1 + w2 * mean2
        base_low = w1 * low1 + w2 * low2
        base_high = w1 * high1 + w2 * high2

        # 情景系数
        if scenario == "乐观":
            factor = 1.10
        elif scenario == "谨慎":
            factor = 0.90
        else:
            factor = 1.00

        scenario_mean = base_mean * factor
        scenario_low = base_low * factor
        scenario_high = base_high * factor

        forecast_df = pd.DataFrame({
            "日期": format_date_series(pd.Series(future_dates)),
            "预测均值": scenario_mean,
            "下界（95%）": scenario_low,
            "上界（95%）": scenario_high,
        })

    st.success(
        f"预测完成！当前情景：**{scenario}**；单变量权重 {w1:.2f}，多特征权重 {w2:.2f}。"
    )

    # Plotly 预测图
    fig = build_forecast_figure(history, forecast_df, scenario, viz_mode)
    st.plotly_chart(fig, use_container_width=True)

    # 预测结果表格 + 下载
    st.markdown("### 📄 预测结果（表格展示）")
    st.dataframe(styled_table(forecast_df), use_container_width=True)

    download_df = forecast_df.copy()
    num_cols = ["预测均值", "下界（95%）", "上界（95%）"]
    download_df[num_cols] = download_df[num_cols].round(2)
    csv_bytes = download_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 下载预测结果（CSV，中文表头，含情景）",
        csv_bytes,
        file_name=f"现金流预测结果_{scenario}情景.csv",
        mime="text/csv",
    )

    # 资金缺口预警与司库调度建议（简单红线规则）
    st.subheader("🚨 资金缺口预警与司库调度建议")

    horizon = min(30, len(forecast_df))
    future_window = forecast_df.head(horizon).copy()
    negatives = future_window["预测均值"] < 0

    if not negatives.any():
        st.success(
            "未来30日预测净现金流整体为正，暂未触发资金缺口预警，可按既定计划稳健推进。"
        )
        st.markdown(
            "- 建议继续关注大额项目支出与回款节奏，保持与银行授信方的沟通畅通；  \n"
            "- 可适度运用短期理财或结构性存款，提高闲置资金收益。"
        )
    else:
        max_streak = 0
        cur_streak = 0
        for is_neg in negatives:
            if is_neg:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0

        first_neg_date = future_window.loc[negatives, "日期"].iloc[0]

        if max_streak >= 3:
            st.error(
                f"未来30日内存在连续 {max_streak} 天净现金流为负，"
                f"首次出现缺口日期约为：{first_neg_date}，需立即启动资金预案。"
            )
            st.markdown(
                "- 建议立即梳理在手货币资金、未使用授信额度和内部资金池可调度空间；  \n"
                "- 对大额资本性支出、低收益项目支出进行节奏重排或暂缓；  \n"
                "- 提前与主要往来银行沟通，锁定短期流动性支持方案（如流动资金贷款、银票池等）；  \n"
                "- 强化应收账款催收和保理等工具运用，缩短资金回笼周期。"
            )
        else:
            st.warning(
                f"未来30日内部分日期净现金流为负，首次出现缺口日期约为：{first_neg_date}，"
                "建议提前统筹安排。"
            )
            st.markdown(
                "- 建议对缺口时点前后的资金收支进行精细化排期，避免集中支出叠加；  \n"
                "- 通过应收账款盘点、加快开票及回款、内部单位互调等方式，增强短期流动性；  \n"
                "- 结合预测结果，必要时可预先锁定部分银行授信备用额度，以防外部环境突变。"
            )

    # 异常检测
    st.subheader("⚠ 历史净现金流异常检测（IQR + Z-score）")
    anomalies_raw = detect_anomalies_combined(history["date"], history[target])
    anomalies_cn = anomalies_to_chinese(anomalies_raw)
    if anomalies_cn.empty:
        st.info("未检测到显著异常点。")
    else:
        st.dataframe(styled_table(anomalies_cn), use_container_width=True)

    # 敏感性分析
    st.subheader("🔍 特征敏感性分析（中文）")
    if multi_features:
        sens_raw = feature_sensitivity_last_window(
            m2, X2[-window_size:], multi_features, ts2
        )
        sens_cn = sensitivity_to_chinese(sens_raw)
        st.dataframe(styled_table(sens_cn), use_container_width=True)
    else:
        st.info("当前数据缺少业务特征列，无法进行敏感性分析。")


if __name__ == "__main__":
    main()
