import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

KMEANS_BASE_URL = "https://mvp-regime-detection-engine-499158301097.europe-west1.run.app"
SUPERVISED_BASE_URL = "https://regime-detection-engine-ptfl-499158301097.europe-west1.run.app"
PORTFOLIO_BASE_URL = "https://regime-detection-engine-ptfl-499158301097.europe-west1.run.app"

KMEANS_PREDICT_URL = f"{KMEANS_BASE_URL}/predict"

PREDICT_LATEST_URL = f"{SUPERVISED_BASE_URL}/predict/latest"
PREDICT_SERIES_URL = f"{SUPERVISED_BASE_URL}/predict/series"
ANALYZE_URL = f"{SUPERVISED_BASE_URL}/analyze"
CUMULATIVE_DATA_URL = f"{SUPERVISED_BASE_URL}/cumulative_data"

PORTFOLIO_URL = f"{PORTFOLIO_BASE_URL}/portfolio"

KMEANS_MODEL = "kmeans_v1"
SUPERVISED_MODEL = "xgb_target4_5d_platt_oof_v1"
PORTFOLIO_MODEL = "rf_target4_5d_platt_oof_v1"

KMEANS_MODEL_INFO = {
    "description": "This model is a K-means clustering model.",
    "date_min": "2008-05-01",
    "date_max": "2026-01-01",
    "label_mapping": {
        "0": "low_vol",
        "1": "interm_vol",
        "2": "high_vol",
    },
}

SUPERVISED_MODEL_INFO = {
    "description": "This model predicts Risk-On / Risk-Off regimes using supervised learning.",
    "date_min": "2007-01-01",
    "date_max": "2026-04-14",
    "label_mapping": {
        "0": "Risk-On",
        "1": "Risk-Off",
    },
}

PORTFOLIO_MODEL_INFO = {
    "description": "This model is used for the regime-switching portfolio backtest.",
    "date_min": "2007-01-01",
    "date_max": "2026-04-14",
    "label_mapping": {
        "0": "Risk-On",
        "1": "Risk-Off",
    },
}

REGIME_DISPLAY = {
    "low_vol": "Low Vol",
    "interm_vol": "Intermediate Vol",
    "high_vol": "High Vol",
    "Risk-On": "Risk-On",
    "Risk-Off": "Risk-Off",
    "risk_on": "Risk-On",
    "risk_off": "Risk-Off",
    "unknown": "Unknown",
}

REGIME_COLORS = {
    "low_vol": "#2E8B57",
    "interm_vol": "#F4A261",
    "high_vol": "#D62828",
    "Risk-On": "#2E8B57",
    "Risk-Off": "#D62828",
    "risk_on": "#2E8B57",
    "risk_off": "#D62828",
    "unknown": "#808080",
}

ASSET_DISPLAY = {
    "^GSPC": "S&P 500",
    "SP500": "S&P 500",
    "TLT": "US 10Y Bonds",
    "US_10Y": "US 10Y Bonds",
    "GC=F": "Gold",
    "Gold": "Gold",
    "BTC-USD": "Bitcoin",
    "BTC": "Bitcoin",
    "PORTFOLIO": "Portfolio",
    "portfolio": "Portfolio",
}

DEFAULT_WEIGHTS = {
    "SP500": 0.5,
    "US_10Y": 0.2,
    "Gold": 0.2,
    "BTC": 0.1,
}

st.set_page_config(
    page_title="Regime Detection Engine",
    page_icon="📈",
    layout="wide",
)


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def format_decimal(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def request_get(url: str, params: dict, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def request_post(url: str, payload: dict, timeout: int = 90) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, ttl=600)
def get_single_prediction(model_selection: str, date_pred: str) -> dict:
    return request_get(
        KMEANS_PREDICT_URL,
        {
            "model_selection": model_selection,
            "date_pred": date_pred,
        },
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_range_prediction(model_selection: str, date_start: str, date_end: str) -> dict:
    return request_get(
        KMEANS_PREDICT_URL,
        {
            "model_selection": model_selection,
            "date_start": date_start,
            "date_end": date_end,
        },
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_latest_prediction(model_selection: str) -> dict:
    return request_get(
        PREDICT_LATEST_URL,
        {
            "model_selection": model_selection,
        },
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_supervised_series(model_selection: str, date_start: str, date_end: str) -> dict:
    return request_get(
        PREDICT_SERIES_URL,
        {
            "model_selection": model_selection,
            "date_start": date_start,
            "date_end": date_end,
        },
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_analyze_data(model_selection: str, date_start: str, date_end: str, weights: dict) -> dict:
    return request_post(
        ANALYZE_URL,
        {
            "model_selection": model_selection,
            "date_start": date_start,
            "date_end": date_end,
            "source": "predicted",
            "weights": weights,
            "custom_tickers": {},
        },
        timeout=120,
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_cumulative_data(model_selection: str, date_start: str, date_end: str, weights: dict) -> dict:
    return request_post(
        CUMULATIVE_DATA_URL,
        {
            "model_selection": model_selection,
            "date_start": date_start,
            "date_end": date_end,
            "weights": weights,
            "custom_tickers": {},
        },
        timeout=120,
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_portfolio_data(
    model_selection: str,
    date_start: str,
    date_end: str,
    alloc_on: dict,
    alloc_off: dict,
) -> dict:
    payload = {
        "model_selection": model_selection,
        "date_start": date_start,
        "date_end": date_end,
        "alloc_on": alloc_on,
        "alloc_off": alloc_off,
        "min_episode_days": 20,
        "transaction_cost": 0.001,
        "name": "Regime Portfolio",
        "benchmark_name": "Buy & Hold",
    }
    return request_post(PORTFOLIO_URL, payload, timeout=300)


def create_single_date_chart(history_df: pd.DataFrame, selected_date: pd.Timestamp, regime_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["prediction_date"],
            y=history_df["vix_level"],
            mode="lines",
            name="VIX",
            line=dict(color="black", width=2),
        )
    )

    selected_row = history_df.loc[history_df["prediction_date"] == selected_date].iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[selected_row["prediction_date"]],
            y=[selected_row["vix_level"]],
            mode="markers",
            name=f"Predicted regime: {REGIME_DISPLAY.get(regime_label, regime_label)}",
            marker=dict(
                size=14,
                color=REGIME_COLORS.get(regime_label, "#808080"),
                line=dict(color="black", width=1),
            ),
        )
    )

    fig.add_vline(
        x=selected_date,
        line_width=2,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        title="VIX 30 Days Before and After Selected Date",
        xaxis_title="Date",
        yaxis_title="VIX Level",
        legend_title="Legend",
        height=500,
    )

    return fig


def create_range_chart(history_df: pd.DataFrame) -> go.Figure:
    regime_to_numeric = {
        "low_vol": 0,
        "interm_vol": 1,
        "high_vol": 2,
    }

    history_df = history_df.copy()
    history_df["regime_numeric"] = history_df["regime_label"].map(regime_to_numeric)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("VIX Evolution", "Regime Evolution"),
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["prediction_date"],
            y=history_df["vix_level"],
            mode="lines",
            name="VIX",
            line=dict(color="black", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["prediction_date"],
            y=history_df["regime_numeric"],
            mode="lines",
            name="Regime",
            line=dict(color="#1f77b4", width=2, shape="hv"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="VIX Level", row=1, col=1)
    fig.update_yaxes(
        title_text="Regime",
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Low Vol", "Intermediate Vol", "High Vol"],
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        title="Market Regime and VIX Over Selected Range",
        showlegend=False,
        xaxis2_title="Date",
    )

    return fig


def create_supervised_series_chart(series_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Risk-Off Probability", "Predicted Regime"),
    )

    fig.add_trace(
        go.Scatter(
            x=series_df["date"],
            y=series_df["proba"],
            mode="lines",
            name="Risk-Off probability",
            line=dict(color="#D62828", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=0.16,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold",
        row=1,
        col=1,
    )

    regime_map = {"Risk-On": 0, "Risk-Off": 1}
    series_df = series_df.copy()
    series_df["regime_numeric"] = series_df["regime_label"].map(regime_map)

    fig.add_trace(
        go.Scatter(
            x=series_df["date"],
            y=series_df["regime_numeric"],
            mode="lines",
            name="Predicted regime",
            line=dict(color="#0B7285", width=2, shape="hv"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Probability", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(
        title_text="Regime",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Risk-On", "Risk-Off"],
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Supervised Risk Regime Signal",
        height=650,
        showlegend=False,
        xaxis2_title="Date",
    )

    return fig


def create_cumulative_chart(cumulative_data: dict, selected_asset: str) -> go.Figure:
    asset_data = cumulative_data["assets"][selected_asset]

    fig = go.Figure()

    for key, label, color in [
        ("buy_hold", "Buy & Hold", "#0B1F3A"),
        ("risk_on_only", "Risk-On only", "#2E8B57"),
        ("risk_off_only", "Risk-Off only", "#D62828"),
    ]:
        series = asset_data[key]
        series_df = pd.DataFrame(series)
        series_df["date"] = pd.to_datetime(series_df["date"])

        fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["value"],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title=f"Cumulative Return by Regime Filter - {ASSET_DISPLAY.get(selected_asset, selected_asset)}",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        legend_title="Strategy",
        height=500,
    )

    return fig


def create_stats_bar_chart(stats_df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()

    for regime in stats_df["regime_label"].unique():
        regime_df = stats_df[stats_df["regime_label"] == regime]

        fig.add_trace(
            go.Bar(
                x=regime_df["name"].map(lambda x: ASSET_DISPLAY.get(x, x)),
                y=regime_df[metric],
                name=REGIME_DISPLAY.get(regime, regime),
            )
        )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} by Asset and Regime",
        xaxis_title="Asset",
        yaxis_title=metric.replace("_", " ").title(),
        barmode="group",
        height=450,
    )

    return fig


def create_weights_chart(weights_data: list) -> go.Figure:
    rows = []

    for item in weights_data:
        date = item["date"]

        for ticker, weight in item["weights"].items():
            rows.append(
                {
                    "date": date,
                    "asset": ASSET_DISPLAY.get(ticker, ticker),
                    "weight": weight,
                }
            )

    weights_df = pd.DataFrame(rows)
    weights_df["date"] = pd.to_datetime(weights_df["date"])

    fig = go.Figure()

    for asset in weights_df["asset"].unique():
        asset_df = weights_df[weights_df["asset"] == asset]

        fig.add_trace(
            go.Scatter(
                x=asset_df["date"],
                y=asset_df["weight"],
                mode="lines",
                stackgroup="one",
                name=asset,
            )
        )

    fig.update_layout(
        title="Portfolio Allocation Weights Over Time",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        legend_title="Asset",
        height=500,
    )

    return fig


def create_portfolio_dashboard_chart(portfolio_data: dict, benchmark_data=None) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            "Cumulative Return",
            "Drawdown",
            "Rolling Sharpe 1Y",
        ),
    )

    regime_signal = portfolio_data.get("regime_signal", [])

    if regime_signal:
        spans = []
        start = None

        for item in regime_signal:
            regime = item.get("regime")
            date = pd.to_datetime(item["date"])

            is_risk_off = regime in ["Risk-Off", "risk_off", 0]

            if is_risk_off and start is None:
                start = date
            elif not is_risk_off and start is not None:
                spans.append((start, date))
                start = None

        if start is not None:
            spans.append((start, pd.to_datetime(regime_signal[-1]["date"])))

        for row in [1, 2, 3]:
            for span_start, span_end in spans:
                fig.add_vrect(
                    x0=span_start,
                    x1=span_end,
                    fillcolor="rgba(214, 40, 40, 0.08)",
                    line_width=0,
                    row=row,
                    col=1,
                )

    p_eq = portfolio_data["equity_curve"]
    p_eq_dates = pd.to_datetime(p_eq["dates"])
    p_cum = p_eq.get("cum_return_pct")

    if p_cum is None:
        p_equity = pd.Series(p_eq["equity"])
        p_cum = (p_equity / p_equity.iloc[0] - 1.0) * 100

    fig.add_trace(
        go.Scatter(
            x=p_eq_dates,
            y=p_cum,
            mode="lines",
            name=portfolio_data.get("name", "Portfolio"),
            line=dict(color="#00d4aa", width=2.5),
        ),
        row=1,
        col=1,
    )

    if benchmark_data and "equity_curve" in benchmark_data:
        b_eq = benchmark_data["equity_curve"]
        b_eq_dates = pd.to_datetime(b_eq["dates"])
        b_cum = b_eq.get("cum_return_pct")

        if b_cum is None:
            b_equity = pd.Series(b_eq["equity"])
            b_cum = (b_equity / b_equity.iloc[0] - 1.0) * 100

        fig.add_trace(
            go.Scatter(
                x=b_eq_dates,
                y=b_cum,
                mode="lines",
                name=benchmark_data.get("name", "Benchmark"),
                line=dict(color="#d4a000", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    p_dd = portfolio_data["drawdown"]
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(p_dd["dates"]),
            y=p_dd["drawdown_pct"],
            mode="lines",
            name="Portfolio drawdown",
            line=dict(color="#00d4aa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.15)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if benchmark_data and "drawdown" in benchmark_data:
        b_dd = benchmark_data["drawdown"]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(b_dd["dates"]),
                y=b_dd["drawdown_pct"],
                mode="lines",
                name="Benchmark drawdown",
                line=dict(color="#d4a000", width=1.2, dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(212, 160, 0, 0.08)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    p_rs = portfolio_data["rolling_sharpe_1y"]
    p_rs_values = [v if v is not None else None for v in p_rs["sharpe"]]
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(p_rs["dates"]),
            y=p_rs_values,
            mode="lines",
            name="Portfolio rolling Sharpe",
            line=dict(color="#00d4aa", width=1.8),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    if benchmark_data and "rolling_sharpe_1y" in benchmark_data:
        b_rs = benchmark_data["rolling_sharpe_1y"]
        b_rs_values = [v if v is not None else None for v in b_rs["sharpe"]]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(b_rs["dates"]),
                y=b_rs_values,
                mode="lines",
                name="Benchmark rolling Sharpe",
                line=dict(color="#d4a000", width=1.2, dash="dash"),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)

    fig.update_yaxes(title_text="Cumulative return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Rolling Sharpe 1Y", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.update_layout(
        title=f"{portfolio_data.get('name', 'Portfolio')} vs {benchmark_data.get('name', 'Benchmark') if benchmark_data else 'Benchmark'}",
        height=950,
        legend_title="Series",
        hovermode="x unified",
    )

    return fig


st.title("Regime Detection Engine")

st.markdown(
    """
    This application displays regime predictions, VIX evolution, predicted asset behavior
    and portfolio strategy outputs using deployed backend APIs.
    """
)

st.sidebar.header("Model Configuration")

st.sidebar.markdown("### Unsupervised model")
st.sidebar.write(f"**Model:** {KMEANS_MODEL}")
st.sidebar.write(f"**Description:** {KMEANS_MODEL_INFO['description']}")
st.sidebar.write("**API:** MVP API")
st.sidebar.write(f"**Available dates:** {KMEANS_MODEL_INFO['date_min']} to {KMEANS_MODEL_INFO['date_max']}")

st.sidebar.markdown("### Supervised model")
st.sidebar.write(f"**Model:** {SUPERVISED_MODEL}")
st.sidebar.write(f"**Description:** {SUPERVISED_MODEL_INFO['description']}")
st.sidebar.write("**API:** Supervised API")
st.sidebar.write(f"**Available dates:** {SUPERVISED_MODEL_INFO['date_min']} to {SUPERVISED_MODEL_INFO['date_max']}")

st.sidebar.markdown("### Portfolio model")
st.sidebar.write(f"**Model:** {PORTFOLIO_MODEL}")
st.sidebar.write(f"**Description:** {PORTFOLIO_MODEL_INFO['description']}")
st.sidebar.write("**API:** Portfolio API")
st.sidebar.write(f"**Available dates:** {PORTFOLIO_MODEL_INFO['date_min']} to {PORTFOLIO_MODEL_INFO['date_max']}")

tab_regime, tab_supervised, tab_predicted, tab_portfolio = st.tabs(
    [
        "Regime Prediction",
        "Risk-Off Signal",
        "Predicted Analysis",
        "Portfolio Dashboard",
    ]
)

with tab_regime:
    st.header("Unsupervised Regime Prediction")

    date_min = pd.to_datetime(KMEANS_MODEL_INFO["date_min"]).date()
    date_max = pd.to_datetime(KMEANS_MODEL_INFO["date_max"]).date()

    mode = st.radio("Select mode", ["Single date", "Date range"], horizontal=True)

    if mode == "Single date":
        with st.form("single_date_form"):
            selected_date = st.date_input(
                "Prediction date",
                value=date_min,
                min_value=date_min,
                max_value=date_max,
            )
            submitted = st.form_submit_button("Predict")

        if submitted:
            selected_date_ts = pd.Timestamp(selected_date)
            selected_date_str = selected_date_ts.strftime("%Y-%m-%d")

            try:
                data = get_single_prediction(KMEANS_MODEL, selected_date_str)

                prediction_df = pd.DataFrame(data["prediction_output"])
                prediction_df["prediction_date"] = pd.to_datetime(prediction_df["prediction_date"])

                row = prediction_df.iloc[0]
                display_regime_label = row["regime_label"]

                st.subheader("Market Regime Prediction")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Predicted regime", REGIME_DISPLAY.get(display_regime_label, display_regime_label))

                with col2:
                    st.metric("Prediction date", row["prediction_date"].strftime("%Y-%m-%d"))

                with col3:
                    st.metric("Model name", data["model_name"])

                with col4:
                    if "vix_level" in row:
                        st.metric("VIX", round(float(row["vix_level"]), 2))
                    else:
                        st.metric("VIX", "N/A")

                window_start = (selected_date_ts - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                window_end = (selected_date_ts + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

                history_data = get_range_prediction(KMEANS_MODEL, window_start, window_end)
                history_df = pd.DataFrame(history_data["prediction_output"])
                history_df["prediction_date"] = pd.to_datetime(history_df["prediction_date"])

                if not history_df.empty and (history_df["prediction_date"] == selected_date_ts).any():
                    st.subheader("VIX Around Selected Date")
                    st.plotly_chart(
                        create_single_date_chart(history_df, selected_date_ts, display_regime_label),
                        use_container_width=True,
                    )
                else:
                    st.warning("No surrounding history available for the selected date.")

                with st.expander("Raw API payload"):
                    st.json(data)

            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    else:
        default_end = min(
            pd.Timestamp(date_max),
            pd.Timestamp(date_min) + pd.Timedelta(days=30),
        ).date()

        with st.form("date_range_form"):
            col_start, col_end = st.columns(2)

            with col_start:
                start_date = st.date_input(
                    "Start date",
                    value=date_min,
                    min_value=date_min,
                    max_value=date_max,
                    key="kmeans_start_date",
                )

            with col_end:
                end_date = st.date_input(
                    "End date",
                    value=default_end,
                    min_value=date_min,
                    max_value=date_max,
                    key="kmeans_end_date",
                )

            submitted = st.form_submit_button("Predict")

        if submitted:
            if start_date > end_date:
                st.error("Start date must be earlier than or equal to end date.")
            else:
                try:
                    data = get_range_prediction(
                        KMEANS_MODEL,
                        pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                        pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                    )

                    history_df = pd.DataFrame(data["prediction_output"])
                    history_df["prediction_date"] = pd.to_datetime(history_df["prediction_date"])

                    st.subheader("Range Analysis")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Start date", pd.Timestamp(start_date).strftime("%Y-%m-%d"))

                    with col2:
                        st.metric("End date", pd.Timestamp(end_date).strftime("%Y-%m-%d"))

                    with col3:
                        st.metric("Observations", len(history_df))

                    with col4:
                        st.metric("Unique regimes", history_df["regime_label"].nunique())

                    st.plotly_chart(create_range_chart(history_df), use_container_width=True)

                    with st.expander("Raw API payload"):
                        st.json(data)

                except requests.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Connection error: {e}")

with tab_supervised:
    st.header("Supervised Risk-Off Signal")

    try:
        latest_data = get_latest_prediction(SUPERVISED_MODEL)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Latest date", latest_data["date"])

        with col2:
            st.metric("Current regime", latest_data["regime"])

        with col3:
            st.metric("Risk-Off probability", format_percent(latest_data["proba"]))

        with col4:
            st.metric("Threshold", format_percent(latest_data["threshold"]))

        with col5:
            st.metric("Confidence", format_decimal(latest_data["confidence"]))

    except requests.HTTPError as e:
        st.error(f"API error while loading latest prediction: {e.response.text}")
    except requests.RequestException as e:
        st.error(f"Connection error while loading latest prediction: {e}")

    date_min = pd.to_datetime(SUPERVISED_MODEL_INFO["date_min"]).date()
    date_max = pd.to_datetime(SUPERVISED_MODEL_INFO["date_max"]).date()

    with st.form("supervised_series_form"):
        col_start, col_end = st.columns(2)

        with col_start:
            start_date = st.date_input(
                "Start date",
                value=pd.to_datetime("2020-01-01").date(),
                min_value=date_min,
                max_value=date_max,
                key="supervised_start_date",
            )

        with col_end:
            end_date = st.date_input(
                "End date",
                value=pd.to_datetime("2025-01-01").date(),
                min_value=date_min,
                max_value=date_max,
                key="supervised_end_date",
            )

        submitted = st.form_submit_button("Load Risk-Off series")

    if submitted:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
        else:
            try:
                with st.spinner("Loading Risk-Off series..."):
                    series_data = get_supervised_series(
                        SUPERVISED_MODEL,
                        pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                        pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                    )

                st.session_state["supervised_series_data"] = series_data
                st.session_state["supervised_series_loaded"] = True

            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    if st.session_state.get("supervised_series_loaded", False):
        series_data = st.session_state["supervised_series_data"]
        series_df = pd.DataFrame(series_data["predictions"])
        series_df["date"] = pd.to_datetime(series_df["date"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Observations", series_data["n_observations"])

        with col2:
            st.metric("Risk-Off days", int((series_df["regime_label"] == "Risk-Off").sum()))

        with col3:
            st.metric("Average probability", format_percent(series_df["proba"].mean()))

        st.plotly_chart(create_supervised_series_chart(series_df), use_container_width=True)

        with st.expander("Raw API payload"):
            st.json(series_data)

with tab_predicted:
    st.header("Predicted Regime Analysis")

    date_min = pd.to_datetime(SUPERVISED_MODEL_INFO["date_min"]).date()
    date_max = pd.to_datetime(SUPERVISED_MODEL_INFO["date_max"]).date()

    with st.form("predicted_analysis_form"):
        col_start, col_end = st.columns(2)

        with col_start:
            start_date = st.date_input(
                "Start date",
                value=pd.to_datetime("2020-01-01").date(),
                min_value=date_min,
                max_value=date_max,
                key="analysis_start_date",
            )

        with col_end:
            end_date = st.date_input(
                "End date",
                value=pd.to_datetime("2025-01-01").date(),
                min_value=date_min,
                max_value=date_max,
                key="analysis_end_date",
            )

        submitted = st.form_submit_button("Run analysis")

    if submitted:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
        else:
            date_start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            date_end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")

            try:
                with st.spinner("Running analysis..."):
                    analyze_data = get_analyze_data(
                        SUPERVISED_MODEL,
                        date_start_str,
                        date_end_str,
                        DEFAULT_WEIGHTS,
                    )

                    cumulative_data = get_cumulative_data(
                        SUPERVISED_MODEL,
                        date_start_str,
                        date_end_str,
                        DEFAULT_WEIGHTS,
                    )

                st.session_state["analyze_data"] = analyze_data
                st.session_state["cumulative_data"] = cumulative_data
                st.session_state["analysis_loaded"] = True

            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {e}")

    if st.session_state.get("analysis_loaded", False):
        analyze_data = st.session_state["analyze_data"]
        cumulative_data = st.session_state["cumulative_data"]

        stats_df = pd.DataFrame(analyze_data["stats"])

        st.subheader("Asset Performance by Predicted Regime")
        st.dataframe(stats_df, use_container_width=True)

        metric = st.selectbox(
            "Select metric to compare",
            ["ann_return", "ann_vol", "sharpe", "hit_rate"],
            index=0,
            key="analysis_metric_select",
        )

        st.plotly_chart(
            create_stats_bar_chart(stats_df, metric),
            use_container_width=True,
        )

        st.subheader("Cumulative Performance by Regime Filter")

        asset_names = list(cumulative_data["assets"].keys())

        selected_asset = st.selectbox(
            "Select asset",
            asset_names,
            format_func=lambda x: ASSET_DISPLAY.get(x, x),
            index=0,
            key="analysis_asset_select",
        )

        st.plotly_chart(
            create_cumulative_chart(cumulative_data, selected_asset),
            use_container_width=True,
        )

        with st.expander("Analyze raw payload"):
            st.json(analyze_data)

        with st.expander("Cumulative raw payload"):
            st.json(cumulative_data)

with tab_portfolio:
    st.header("Portfolio Dashboard")

    date_min = pd.to_datetime(PORTFOLIO_MODEL_INFO["date_min"]).date()
    date_max = pd.to_datetime(PORTFOLIO_MODEL_INFO["date_max"]).date()

    with st.form("portfolio_form"):
        col_start, col_end = st.columns(2)

        with col_start:
            start_date = st.date_input(
                "Start date",
                value=pd.to_datetime("2012-03-09").date(),
                min_value=date_min,
                max_value=date_max,
                key="portfolio_start_date",
            )

        with col_end:
            end_date = st.date_input(
                "End date",
                value=pd.to_datetime("2024-01-18").date(),
                min_value=date_min,
                max_value=date_max,
                key="portfolio_end_date",
            )

        st.subheader("Risk-On Allocation")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sp500_on = st.number_input("SP500", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="sp500_on")
        with col2:
            us10y_on = st.number_input("US_10Y", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="us10y_on")
        with col3:
            gold_on = st.number_input("Gold", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="gold_on")
        with col4:
            btc_on = st.number_input("BTC", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="btc_on")

        st.subheader("Risk-Off Allocation")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sp500_off = st.number_input("SP500 ", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="sp500_off")
        with col2:
            us10y_off = st.number_input("US_10Y ", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="us10y_off")
        with col3:
            gold_off = st.number_input("Gold ", min_value=0.0, max_value=1.0, value=0.30, step=0.01, key="gold_off")
        with col4:
            btc_off = st.number_input("BTC ", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="btc_off")

        submitted = st.form_submit_button("Run portfolio backtest")

    if submitted:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
        else:
            alloc_on = {
                "SP500": sp500_on,
                "US_10Y": us10y_on,
                "Gold": gold_on,
                "BTC": btc_on,
            }

            alloc_off = {
                "SP500": sp500_off,
                "US_10Y": us10y_off,
                "Gold": gold_off,
                "BTC": btc_off,
            }

            sum_on = sum(alloc_on.values())
            sum_off = sum(alloc_off.values())

            if abs(sum_on - 1.0) > 0.001:
                st.error(f"Risk-On allocation must sum to 1.00. Current sum: {sum_on:.2f}")
            elif abs(sum_off - 1.0) > 0.001:
                st.error(f"Risk-Off allocation must sum to 1.00. Current sum: {sum_off:.2f}")
            else:
                try:
                    with st.spinner("Running portfolio backtest..."):
                        portfolio_dashboard = get_portfolio_data(
                            PORTFOLIO_MODEL,
                            pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                            pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                            alloc_on,
                            alloc_off,
                        )

                    st.session_state["portfolio_dashboard"] = portfolio_dashboard
                    st.session_state["portfolio_loaded"] = True

                except requests.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Connection error: {e}")

    if st.session_state.get("portfolio_loaded", False):
        portfolio_dashboard = st.session_state["portfolio_dashboard"]

        portfolio_data = portfolio_dashboard["portfolio"]
        benchmark_data = portfolio_dashboard.get("benchmark")

        st.subheader(portfolio_data["name"])

        metrics = portfolio_data.get("metrics_formatted", {})

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total return", metrics.get("total_return_pct", "N/A"))

        with col2:
            st.metric("Ann. return", metrics.get("ann_return_pct", "N/A"))

        with col3:
            st.metric("Ann. vol", metrics.get("ann_vol_pct", "N/A"))

        with col4:
            st.metric("Sharpe", metrics.get("sharpe", "N/A"))

        with col5:
            st.metric("Max drawdown", metrics.get("max_drawdown_pct", "N/A"))

        with col6:
            st.metric("Risk-off time", metrics.get("%_time_risk_off", "N/A"))

        st.plotly_chart(
            create_portfolio_dashboard_chart(portfolio_data, benchmark_data),
            use_container_width=True,
        )

        if "weights" in portfolio_data and portfolio_data["weights"]:
            st.plotly_chart(
                create_weights_chart(portfolio_data["weights"]),
                use_container_width=True,
            )

        with st.expander("Portfolio raw payload"):
            st.json(portfolio_dashboard)
