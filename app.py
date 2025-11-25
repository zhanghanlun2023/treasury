def build_forecast_figure(
    history,
    forecast_df,
    scenario_name,
    viz_mode="标准模式",
):
    dates_hist = format_date_series(history["date"])
    dates_fut = forecast_df["日期"]
    hist_values = history["net_cash_flow"]

    # ========== 🔥 主题样式设定（蓝金渐变） ==========
    gradient_bg = """
    linear-gradient(135deg, rgba(0,85,164,0.15) 0%, rgba(207,175,112,0.12) 100%)
    """

    # 颜色主题
    if viz_mode == "暗黑模式":
        template = "plotly_dark"
        bg_color = "#0d1117"
        plot_bg = "#0d1117"
        line_hist_color = "#66CCFF"
        line_pred_color = "#FFD700"
        band_color = "rgba(255,215,0,0.2)"
        zero_line_color = "#FF4B4B"
    else:
        template = "plotly_white"
        bg_color = "#F5F7FA"
        plot_bg = "white"
        line_hist_color = BLUE
        line_pred_color = GOLD
        band_color = "rgba(0, 85, 164, 0.20)"
        zero_line_color = "red"

    # ============================
    #    图表开始构建
    # ============================
    fig = go.Figure()

    # ========== 🔵 历史数据线 ==========
    fig.add_trace(
        go.Scatter(
            x=dates_hist,
            y=hist_values,
            mode="lines",
            name="历史净现金流",
            line=dict(color=line_hist_color, width=3),
            hovertemplate="<b>日期</b>: %{x}<br><b>净现金流</b>: %{y:,.2f}<extra></extra>",
        )
    )

    # ========== 🟡 预测均值线（虚线，带动画） ==========
    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["预测均值"],
            mode="lines",
            name=f"{scenario_name}情景预测",
            line=dict(color=line_pred_color, width=3, dash="dash"),
            hovertemplate="<b>预测日期</b>: %{x}<br><b>预测值</b>: %{y:,.2f}"
                          "<br><b>AI 风险提示</b>: %{customdata}<extra></extra>",
            customdata=[
                "⚠ 可能出现现金缺口" if v < 0 else "✓ 现金流健康"
                for v in forecast_df["预测均值"]
            ],
        )
    )

    # ========== 🌀 置信区间（渐变透明带） ==========
    fig.add_trace(
        go.Scatter(
            x=dates_fut,
            y=forecast_df["上界（95%）"],
            mode="lines",
            line=dict(width=0),
            name="上界",
            showlegend=False,
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

    # ========== 🔴 零线（资金缺口预警线） ==========
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=zero_line_color,
        annotation_text="缺口预警(0)",
        annotation_position="top left",
        annotation_font=dict(color=zero_line_color),
    )

    # ========== 🟦 图表布局设置（大屏风格） ==========
    fig.update_layout(
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        title=dict(
            text=f"AI 现金流预测大屏（情景模式：{scenario_name}）",
            x=0.5,
            font=dict(size=22),
        ),
        xaxis=dict(
            title="日期",
            tickangle=-45,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title="净现金流（元）",
            gridcolor="rgba(0,0,0,0.08)",
        ),
    )

    # ========== ✨ 添加大屏渐变背景（真正炫酷点） ==========
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                fillcolor="rgba(0,0,0,0)",
                layer="below",
                line=dict(width=0),
            )
        ]
    )

    return fig
