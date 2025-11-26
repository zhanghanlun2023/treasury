# app.py
# AI 赋能司库：科技玻璃风 · 现金流预测 & 汇率风险 & 资金池模拟器

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
# 一、全局科技玻璃 UI 主题（亮色 + 科技蓝）
# ============================================================

PRIMARY_BLUE = "#2563EB"   # 科技蓝
ACCENT_CYAN = "#06B6D4"    # 青色高光
ACCENT_GOLD = "#CFAF70"    # 金色点缀
BG_MAIN = "#F3F6FB"        # 页面主背景
TEXT_MAIN = "#0F172A"      # 深色正文
TEXT_SUB = "#6B7280"       # 次要文字

BASE_CSS = f"""
<style>
body {{
    background: radial-gradient(circle at top left, #E0EAFF 0, #F9FAFB 40%, #EEF2FF 100%);
    color: {TEXT_MAIN};
    font-family: "Microsoft YaHei", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_MAIN} !important;
    font-family: "Microsoft YaHei", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.big-number {{
    font-size: 28px;
    font-weight: 700;
    color: {PRIMARY_BLUE};
    text-align: center;
}}

.big-number-gold {{
    font-size: 28px;
    font-weight: 700;
    color: {ACCENT_GOLD};
    text-align: center;
}}

.card {{
    background: rgba(255, 255, 255, 0.82);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    padding: 16px 18px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}}

.card-title {{
    font-size: 14px;
    color: {TEXT_SUB};
    text-align: center;
    margin-top: 8px;
}}

.card-sub {{
    font-size: 12px;
    color: #9CA3AF;
    text-align: center;
    margin-top: 4px;
}}

.block-title {{
    font-size: 18px;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-top: 4px;
    margin-bottom: 8px;
}}

hr {{
    border: none;
    border-top: 1px solid rgba(209, 213, 219, 0.9);
    margin: 12px 0 18px 0;
}}

.home-hero {{
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 32px 28px;
    background:
        radial-gradient(circle at 0% 0%, rgba(59,130,246,0.25) 0, transparent 45%),
        radial-gradient(circle at 100% 0%, rgba(56,189,248,0.20) 0, transparent 50%),
        radial-gradient(circle at 50% 100%, rgba(250,204,21,0.18) 0, transparent 50%),
        #FFFFFFCC;
    box-shadow: 0 24px 60px rgba(15,23,42,0.18);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}}

.home-hero-title {{
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
    color: {TEXT_MAIN};
}}

.home-hero-sub {{
    font-size: 17px;
    color: {TEXT_SUB};
    margin-bottom: 16px;
}}

.pulse-dot {{
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: #22C55E;
    box-shadow: 0 0 14px rgba(34,197,94,0.9);
}}

@keyframes float {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
    100% {{ transform: translateY(0px); }}
}}

.float-card {{
    animation: float 4s ease-in-out infinite;
}}

.sidebar-header {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT_MAIN};
    margin-bottom: 6px;
    margin-top: 12px;
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
    """亮色科技风表格样式"""
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

    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#EFF4FF"),
                    ("color", "#111827"),
                    ("font-weight", "600"),
                    ("border-bottom", "1px solid #CBD5F5"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#111827"),
                    ("border-bottom", "1px solid #E5E7EB"),
                ],
            },
        ]
    )

    return styler


# ============================================================
# 三、数据生成 & 预处理
# ============================================================

def generate_synthetic_data(n_days: int = 730) -> pd.DataFrame:
    """模拟一组带季节性、趋势和噪声的现金流数据"""
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
# 四、LSTM & 简单模型
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


def build_simple_dropout_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
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
        raise ValueError("样本量过少，无法训练 LSTM 模型，请保证数据至少有约 100 天。")

    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(patience=5, restore_best_weights=True)

    # 尝试 LSTM，如失败则回退到简单模型
    try:
        model = build_lstm_model((window, X_seq.shape[2]))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0
        )
    except Exception:
        model = build_simple_dropout_model((window, X_seq.shape[2]))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max(5, epochs // 2),
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
# 七、Plotly 图表 & AI 自动点评（亮色科技风）
# ============================================================

def build_forecast_figure(
    history,
    forecast_df,
    scenario_name,
    viz_mode="标准模式",
):
    template = "plotly_white"

    neon_blue = "#2563EB"
    laser_purple = "#8B5CF6"
    band_color = "rgba(37, 99, 235, 0.12)"
    zero_line_color = "#DC2626"

    dates_hist = format_date_series(history["date"])
    dates_fut = forecast_df["日期"]
    hist_values = history["net_cash_flow"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates_hist,
            y=hist_values,
            mode="lines",
            name="历史净现金流",
            line=dict(color=neon_blue, width=3),
            hovertemplate="<b>日期</b>: %{x}<br><b>净现金流</b>: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["预测均值"],
            mode="lines",
            name=f"{scenario_name}情景预测",
            line=dict(color=laser_purple, width=3, dash="dash"),
            hovertemplate="<b>预测日期</b>: %{x}<br>"
                          "<b>预测值</b>: %{y:,.2f}<br>"
                          "<b>AI 风险提示</b>: %{customdata}"
                          "<extra></extra>",
            customdata=[
                "⚠️ 可能出现现金缺口" if v < 0 else "✓ 现金流健康"
                for v in forecast_df["预测均值"]
            ],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["上界（95%）"],
            mode="lines",
            line=dict(width=0),
            name="上界",
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
            hoverinfo="skip",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=zero_line_color,
        annotation_text="缺口预警(0)",
        annotation_position="top left",
        annotation_font=dict(color=zero_line_color),
    )

    fig.update_layout(
        template=template,
        paper_bgcolor=BG_MAIN,
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        title=dict(
            text=f"🔮 科技玻璃 · AI 现金流预测（情景：{scenario_name}）",
            x=0.5,
            font=dict(size=24, color=TEXT_MAIN),
        ),
        xaxis=dict(
            title="日期",
            tickangle=-45,
            gridcolor="rgba(209,213,219,0.7)",
            color="#374151",
        ),
        yaxis=dict(
            title="净现金流（元）",
            gridcolor="rgba(209,213,219,0.7)",
            color="#374151",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#4B5563"),
        ),
    )

    fig.update_xaxes(showline=True, linewidth=1.2, linecolor="rgba(148,163,184,0.9)")
    fig.update_yaxes(showline=True, linewidth=1.2, linecolor="rgba(148,163,184,0.9)")

    return fig


def generate_ai_commentary(forecast_df: pd.DataFrame, scenario: str):
    vals = forecast_df["预测均值"].values
    mean_val = float(vals.mean())
    min_val = float(vals.min())
    max_val = float(vals.max())
    std_val = float(vals.std(ddof=1))
    neg_ratio = float((vals < 0).mean())
    first_neg_date = None
    if np.any(vals < 0):
        first_neg_date = forecast_df.loc[forecast_df["预测均值"] < 0, "日期"].iloc[0]

    lines = []

    if scenario == "乐观":
        lines.append("当前为【乐观情景】，假设收入兑现度较高、支出执行较为审慎，预测结果整体略向上偏离基准。")
    elif scenario == "谨慎":
        lines.append("当前为【谨慎情景】，在收入打折、支出略有提前的假设下，对未来现金流作保守估计。")
    else:
        lines.append("当前为【中性情景】，在既定预算与历史趋势假设下，对未来现金流进行基准预测。")

    if mean_val >= 0:
        lines.append(f"从均值看，预测区间内日均净现金流约为 **{mean_val:,.0f} 元**，整体处于可控区间。")
    else:
        lines.append(f"从均值看，预测区间内日均净现金流约为 **{mean_val:,.0f} 元**，呈一定资金净流出态势。")

    if std_val < abs(mean_val) * 0.3:
        lines.append("现金流波动率相对温和，说明收支节奏较为平滑，有利于司库做中短期资金统筹。")
    else:
        lines.append("现金流波动率偏高，建议围绕大额收支时点开展专项排期和“日计划”管理。")

    if neg_ratio == 0:
        lines.append("预测区间内未出现净现金流为负的时点，短期资金安全边际较高，可在风险可控前提下适度提高资金使用效率。")
    elif neg_ratio < 0.3:
        lines.append(
            f"约有 {neg_ratio*100:.1f}% 的预测日期净现金流为负，"
            f"首次缺口预计在 **{first_neg_date}**，建议提前准备流动性备份方案。"
        )
    else:
        lines.append(
            f"预测期内有超过 {neg_ratio*100:.1f}% 的日期存在资金缺口风险，"
            f"且最低值下探至 **{min_val:,.0f} 元**，需要从压降支出、加快回款和银行授信等多维度协同化解。"
        )

    if min_val < 0 and abs(min_val) > abs(mean_val) * 2:
        lines.append("极端情形下的最小净现金流偏低，存在尾部风险，建议结合压力测试场景，制定应急资金预案。")
    else:
        lines.append("整体尾部风险在可接受范围内，可结合年度资金计划动态滚动调整。")

    return lines


# ============================================================
# 八、汇率风险监控（模拟）
# ============================================================

def render_fx_risk_page():
    st.subheader("💱 汇率风险监控（情景模拟）")

    col1, col2 = st.columns(2)
    with col1:
        base_ccy = st.selectbox("记账本位币", ["CNY", "USD", "EUR"], index=0)
        fx_pair = st.selectbox("汇率对（模拟）", ["USD/CNY", "EUR/CNY", "USD/ZAR"], index=0)
        exposure = st.number_input("外币敞口金额（例如：应收 USD 金额）", value=5_000_000.0, step=100_000.0)

    with col2:
        spot = st.number_input("当前即期汇率（例如 USD/CNY）", value=7.20, step=0.01)
        vol_annual = st.slider("年化波动率（%）", 5.0, 40.0, 15.0, step=1.0)
        horizon_days = st.slider("风险评估期限（天）", 10, 180, 60, step=10)
        n_sims = st.slider("模拟路径条数", 200, 2000, 800, step=200)

    run = st.button("开始汇率模拟")

    if not run:
        st.info("设置完参数后，点击【开始汇率模拟】。")
        return

    with st.spinner("正在进行汇率路径模拟（GBM 模型）..."):
        dt = horizon_days / 252.0
        sigma = vol_annual / 100.0
        mu = 0.0

        z = np.random.randn(n_sims)
        rates = spot * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        values_base = exposure * rates

        mean_val = float(values_base.mean())
        p95 = float(np.percentile(values_base, 95))
        p5 = float(np.percentile(values_base, 5))
        var_95 = mean_val - p5

    st.success("模拟完成！以下结果仅为演示用途，可用于司库汇率风险沟通。")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f"<div class='card'><div class='big-number'>{mean_val:,.0f}</div>"
            f"<div class='card-title'>预期本位币价值</div></div>",
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"<div class='card'><div class='big-number'>{p5:,.0f}</div>"
            f"<div class='card-title'>5% 分位数（不利情形）</div></div>",
            unsafe_allow_html=True
        )
    with col_c:
        st.markdown(
            f"<div class='card'><div class='big-number'>{p95:,.0f}</div>"
            f"<div class='card-title'>95% 分位数（有利情形）</div></div>",
            unsafe_allow_html=True
        )
    with col_d:
        st.markdown(
            f"<div class='card'><div class='big-number-gold'>{var_95:,.0f}</div>"
            f"<div class='card-title'>95% VaR（损失指标）</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### 📈 汇率敞口价值分布（模拟结果）")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values_base,
            nbinsx=40,
            name="本位币价值分布",
            marker=dict(color="#38BDF8"),
            opacity=0.85,
        )
    )
    fig.add_vline(
        x=mean_val, line_color="#FACC15", line_dash="dash",
        annotation_text="均值", annotation_position="top right"
    )
    fig.add_vline(
        x=p5, line_color="#F97316", line_dash="dot",
        annotation_text="5% 分位数", annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BG_MAIN,
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="未来某一日的本位币价值",
        yaxis_title="模拟频数",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💡 管理含义（示例）：")
    st.markdown(
        f"- 在当前参数下，{horizon_days} 天内该笔外币敞口在 **95% 置信度** 下，最大不利变动约为 **{var_95:,.0f} {base_ccy}**；  \n"
        f"- 可结合外汇套保工具（远期、掉期、期权）以及自然对冲安排，控制敞口在集团司库授权范围内；  \n"
        "- 建议对大额汇率敏感项目定期滚动更新类似模拟结果，用于向管理层汇报。"
    )


# ============================================================
# 九、资金池模拟器
# ============================================================

def render_pool_simulator_page():
    st.subheader("🏦 集团资金池模拟器（总部 + 子公司）")

    st.markdown(
        "通过简单参数设置，模拟“总部司库 + 子公司 A/B”的资金集中效果，"
        "用于演示内部银行 / 资金池机制对资金使用效率的改善。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        hq_cash = st.number_input("总部当前现金余额", value=50_000_000.0, step=1_000_000.0)
        hq_min = st.number_input("总部安全备付线", value=20_000_000.0, step=1_000_000.0)
    with col2:
        a_cash = st.number_input("子公司 A 当前现金", value=15_000_000.0, step=500_000.0)
        a_min = st.number_input("子公司 A 安全备付线", value=5_000_000.0, step=500_000.0)
    with col3:
        b_cash = st.number_input("子公司 B 当前现金", value=8_000_000.0, step=500_000.0)
        b_min = st.number_input("子公司 B 安全备付线", value=3_000_000.0, step=500_000.0)

    target_hq_ratio = st.slider("目标资金集中度（总部占集团货币资金比例）", 0.3, 0.9, 0.6, step=0.05)

    run = st.button("模拟资金归集与下拨方案")

    if not run:
        st.info("设置完参数后，点击【模拟资金归集与下拨方案】。")
        return

    total_cash = hq_cash + a_cash + b_cash
    target_hq_cash = total_cash * target_hq_ratio

    a_surplus = max(0.0, a_cash - a_min)
    b_surplus = max(0.0, b_cash - b_min)

    collect_from_a = 0.0
    collect_from_b = 0.0

    need_for_hq = max(0.0, target_hq_cash - hq_cash)
    if need_for_hq > 0:
        from_a = min(a_surplus, need_for_hq)
        collect_from_a = from_a
        need_for_hq -= from_a

    if need_for_hq > 0:
        from_b = min(b_surplus, need_for_hq)
        collect_from_b = from_b
        need_for_hq -= from_b

    hq_after = hq_cash + collect_from_a + collect_from_b
    a_after = a_cash - collect_from_a
    b_after = b_cash - collect_from_b

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"<div class='card'><div class='big-number'>{total_cash:,.0f}</div>"
            f"<div class='card-title'>集团现金总量</div></div>",
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"<div class='card'><div class='big-number'>{hq_after/total_cash:,.1%}</div>"
            f"<div class='card-title'>归集后总部资金占比</div></div>",
            unsafe_allow_html=True
        )
    with col_c:
        st.markdown(
            f"<div class='card'><div class='big-number-gold'>{target_hq_ratio:.0%}</div>"
            f"<div class='card-title'>目标集中度</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### 📊 归集前后资金分布对比")

    fig = go.Figure()
    entities = ["总部司库", "子公司A", "子公司B"]
    before_vals = [hq_cash, a_cash, b_cash]
    after_vals = [hq_after, a_after, b_after]

    fig.add_trace(go.Bar(x=entities, y=before_vals, name="归集前", marker_color="#93C5FD"))
    fig.add_trace(go.Bar(x=entities, y=after_vals, name="归集后", marker_color="#34D399"))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        paper_bgcolor=BG_MAIN,
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_title="现金余额（元）",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📌 资金归集方案（示例）：")
    st.markdown(
        f"- 建议子公司 A 通过内部资金池向总部归集 **{collect_from_a:,.0f} 元**；  \n"
        f"- 建议子公司 B 通过内部资金池向总部归集 **{collect_from_b:,.0f} 元**；  \n"
        "- 归集后，子公司仍保留各自安全备付线以上资金，用于日常运营；  \n"
        "- 总部集中后的资金可统一统筹偿债、投资和理财，提高集团整体资金使用效率。"
    )


# ============================================================
# 十、首页：科技光效 + 曲线 Hero SVG 动画背景
# ============================================================

def render_home_page():
    # ====== 动画光效 & SVG 曲线背景 CSS ======
    st.markdown(
        """
        <style>

        /* 🌟 光束动画（Neon Beam） */
        @keyframes neon-beam {
            0%   { opacity: 0.0; transform: translateX(-240px) translateY(-80px) rotate(12deg); }
            50%  { opacity: 0.38; transform: translateX(0px) translateY(0px) rotate(12deg); }
            100% { opacity: 0.0; transform: translateX(240px) translateY(80px) rotate(12deg); }
        }

        .neon-beam-layer {
            position:absolute;
            left:0;
            right:0;
            top:0;
            bottom:0;
            background: linear-gradient(120deg, rgba(59,130,246,0.25), rgba(56,189,248,0.20), rgba(250,204,21,0.18));
            filter: blur(36px);
            animation: neon-beam 7s ease-in-out infinite;
            pointer-events:none;
        }

        /* 🌊 扩散光波（Glow Wave） */
        @keyframes glow-wave {
            0% { opacity:0.18; transform:scale(1); }
            50% { opacity:0.32; transform:scale(1.06); }
            100% { opacity:0.18; transform:scale(1); }
        }

        .glow-wave {
            position:absolute;
            width:180%;
            height:180%;
            top:-40%;
            left:-40%;
            background: radial-gradient(circle, rgba(147,197,253,0.28), rgba(255,255,255,0));
            animation: glow-wave 5.5s linear infinite;
            filter: blur(14px);
            pointer-events:none;
        }

        /* 📈 曲线 SVG 背景容器 */
        .hero-svg-container {
            position:absolute;
            right:-40px;
            bottom:-40px;
            width:420px;
            height:220px;
            opacity:0.9;
            pointer-events:none;
        }

        /* 线条描边动画 */
        @keyframes path-draw {
            0%   { stroke-dashoffset: 480; opacity: 0.0; }
            20%  { opacity: 1.0; }
            60%  { stroke-dashoffset: 0;  opacity: 1.0; }
            100% { stroke-dashoffset: 0;  opacity: 0.35; }
        }

        .curve-path {
            stroke-dasharray: 480;
            stroke-dashoffset: 480;
            animation: path-draw 6s ease-in-out infinite;
        }

        /* 玻璃卡片浮动动画 */
        @keyframes slow-float {
            0%   { transform: translateY(0px); }
            50%  { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }

        .float-card {
            animation: slow-float 4.5s ease-in-out infinite;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # ====== 首页内容区（带 SVG 背景） ======
    st.markdown(
        """
        <div class="home-hero" style="position:relative; overflow:hidden; border-radius:24px;">

            <!-- 动态光效层 -->
            <div class="neon-beam-layer"></div>
            <div class="glow-wave"></div>

            <!-- 曲线 Hero SVG 动画背景 -->
            <div class="hero-svg-container">
                <svg viewBox="0 0 400 220" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- 渐变定义 -->
                    <defs>
                        <linearGradient id="gradLine" x1="0" y1="0" x2="400" y2="0">
                            <stop offset="0%" stop-color="#38BDF8" />
                            <stop offset="45%" stop-color="#6366F1" />
                            <stop offset="100%" stop-color="#FACC15" />
                        </linearGradient>
                        <linearGradient id="gradFill" x1="0" y1="0" x2="0" y2="220">
                            <stop offset="0%" stop-color="rgba(129,140,248,0.35)" />
                            <stop offset="100%" stop-color="rgba(255,255,255,0)" />
                        </linearGradient>
                    </defs>

                    <!-- 背景淡网格 -->
                    <g opacity="0.25">
                        <path d="M0 40H400" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M0 80H400" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M0 120H400" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M0 160H400" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M0 200H400" stroke="#E5E7EB" stroke-width="0.5"/>

                        <path d="M50 0V220" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M150 0V220" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M250 0V220" stroke="#E5E7EB" stroke-width="0.5"/>
                        <path d="M350 0V220" stroke="#E5E7EB" stroke-width="0.5"/>
                    </g>

                    <!-- 填充区域 -->
                    <path d="M10 170
                             C 80 130, 120 110, 170 100
                             C 230 88, 270 70, 330 60
                             C 365 55, 390 65, 390 65
                             L 390 220 L 10 220 Z"
                          fill="url(#gradFill)" opacity="0.50"/>

                    <!-- 主曲线 -->
                    <path class="curve-path"
                          d="M10 170
                             C 80 130, 120 110, 170 100
                             C 230 88, 270 70, 330 60
                             C 365 55, 390 65, 390 65"
                          stroke="url(#gradLine)"
                          stroke-width="3.0"
                          stroke-linecap="round"
                          fill="none"/>

                    <!-- 高亮端点 -->
                    <circle cx="390" cy="65" r="4.5" fill="#FACC15" />
                    <circle cx="10" cy="170" r="4" fill="#38BDF8" />
                </svg>
            </div>

            <!-- 顶部小标签 -->
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;position:relative;z-index:2;">
                <div class="pulse-dot"></div>
                <div style="font-size:13px;color:#6B7280;">Financial AI · Treasury Intelligence</div>
            </div>

            <!-- 主标题 -->
            <div class="home-hero-title" style="font-size:36px; font-weight:700; color:#0F172A; position:relative;z-index:2;">
                科技赋能财务 · 智能化司库中枢
            </div>

            <!-- 副标题 -->
            <div class="home-hero-sub" style="font-size:18px; color:#475569; position:relative;z-index:2;">
                一个整合 <span style="color:#2563EB;font-weight:600;">现金流预测</span>、
                <span style="color:#2563EB;font-weight:600;">汇率风险管理</span> 与
                <span style="color:#2563EB;font-weight:600;">集团资金池调度</span> 的
                智能财务平台，支撑资金统筹、风险预警与跨境业务的数字化决策。
            </div>

            <!-- 三个模块卡片 -->
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:22px;position:relative;z-index:2;">

                <div class="card float-card" style="flex:1;min-width:240px;">
                    <div class="card-title" style="font-size:15px;font-weight:600;color:#1F2937;">
                        模块一：AI 现金流预测
                    </div>
                    <div class="card-sub" style="font-size:13px;">
                        通过 LSTM + MC Dropout 生成带置信区间的现金流预测，
                        提供情景分析、资金缺口预警与自动化点评。
                    </div>
                </div>

                <div class="card float-card" style="flex:1;min-width:240px;">
                    <div class="card-title" style="font-size:15px;font-weight:600;color:#1F2937;">
                        模块二：汇率风险监控
                    </div>
                    <div class="card-sub" style="font-size:13px;">
                        采用 GBM 模型模拟外币敞口变化，输出 VaR 指标，
                        支持海外项目与跨境结算的汇率风险管理。
                    </div>
                </div>

                <div class="card float-card" style="flex:1;min-width:240px;">
                    <div class="card-title" style="font-size:15px;font-weight:600;color:#1F2937;">
                        模块三：集团资金池调度
                    </div>
                    <div class="card-sub" style="font-size:13px;">
                        模拟总部与子公司之间的资金归集与下拨流程，
                        演示内部银行和统一司库体系的资金统筹能力。
                    </div>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🚀 使用说明（建议从左侧切换模块）")
    st.markdown(
        """
        - **首页**：展示系统概览、科技动画效果与模块介绍。  
        - **现金流预测主面板**：支持 CSV 上传、AI 预测、置信区间、风险提示等。  
        - **汇率风险监控**：模拟外币敞口价值分布，提供 VaR 和风险解读。  
        - **资金池模拟器**：演示集团式资金集中管理与内部银行机制。  
        - 所有模块均可用于方案汇报、司库建设展示及内部交流。
        """
    )


# ============================================================
# 十一、现金流预测主面板
# ============================================================

def render_cashflow_page():
    st.subheader("📊 资金监控大屏 Dashboard（科技玻璃版）")

    st.sidebar.markdown("<div class='sidebar-header'>⚙ 数据与模型参数（现金流预测）</div>", unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("📤 上传现金流 CSV（含 date 列）", type=["csv"])
    use_synthetic = st.sidebar.checkbox(
        "使用系统模拟数据（忽略上传文件）",
        value=(uploaded_file is None)
    )

    window_size = st.sidebar.slider("时间窗口长度（天）", 30, 120, 60, step=5)
    forecast_days = st.sidebar.slider("预测天数", 7, 180, 90, step=7)
    epochs = st.sidebar.slider("训练轮数（Epoch）", 5, 50, 20, step=5)
    n_samples = st.sidebar.slider("Monte-Carlo Dropout 次数", 10, 100, 30, step=10)

    scenario = st.sidebar.radio(
        "情景模式",
        ["谨慎", "中性", "乐观"],
        index=1,
        help="谨慎：在预测基础上下调 10%；乐观：在预测基础上上调 10%。"
    )

    run_button = st.sidebar.button("🚀 开始训练与预测")

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

    # 顶部指标卡片
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

    st.markdown("#### 💡 AI 初步判断")
    if avg7 < 0:
        st.error("未来短期净现金流偏弱，建议提前统筹资金调度、压降支出并加快回款。")
    else:
        st.success("未来短期净现金流整体平稳偏正，资金安全边际较为充足，可稳步推进既定经营计划。")

    # 数据预览
    st.markdown("### 📁 数据预览（中文表头）")
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
    try:
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
    except ValueError as e:
        st.error(f"训练失败：{e}")
        return

    history = hist2.copy()

    # 集成预测 + 情景
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
        w2 = inv2 / (inv1 + inv2)

        last_date = history["date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

        base_mean = w1 * mean1 + w2 * mean2
        base_low = w1 * low1 + w2 * low2
        base_high = w1 * high1 + w2 * high2

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

    # 图表
    fig = build_forecast_figure(history, forecast_df, scenario, "亮色模式")
    st.plotly_chart(fig, use_container_width=True)

    # AI 自动点评
    st.markdown("### 🧠 AI 司库自动点评")
    comments = generate_ai_commentary(forecast_df, scenario)
    for line in comments:
        st.markdown(f"- {line}")

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

    # 资金缺口预警
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


# ============================================================
# 主入口：多模块路由
# ============================================================

def main():
    st.set_page_config(
        page_title="AI 赋能司库：科技玻璃 · 现金流预测系统",
        layout="wide"
    )

    col_logo, col_title, col_mode = st.columns([1, 4, 2])

    with col_logo:
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=70)
        else:
            st.markdown(
                f"""
                <div style='width:70px;height:70px;border-radius:18px;
                background:linear-gradient(135deg, #3B82F6 0%, #22C55E 40%, #FACC15 100%);
                display:flex;align-items:center;justify-content:center;
                box-shadow:0 12px 30px rgba(37,99,235,0.45);'>
                    <span style='color:white;font-weight:bold;font-size:18px;'>AI</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_title:
        st.markdown(
            f"""
            <h1>AI 赋能司库 · 科技玻璃财务驾驶舱</h1>
            <h4 style="color:{TEXT_SUB};margin-top:-8px;">
                现金流预测 × 汇率风险 × 资金池模拟 × 场景化决策支持
            </h4>
            """,
            unsafe_allow_html=True
        )

    with col_mode:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">当前版本</div>
                <div class="big-number-gold">Treasury · Beta</div>
                <div class="card-sub">适用于内部交流与方案展示</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.sidebar.markdown("<div class='sidebar-header'>🧭 功能模块</div>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "",
        ["首页", "现金流预测主面板", "汇率风险监控", "资金池模拟器"],
        index=0
    )

    if page == "首页":
        render_home_page()
    elif page == "现金流预测主面板":
        render_cashflow_page()
    elif page == "汇率风险监控":
        render_fx_risk_page()
    elif page == "资金池模拟器":
        render_pool_simulator_page()


if __name__ == "__main__":
    main()
