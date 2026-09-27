import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 UAC Care Load & Placement Demand Forecast")

st.caption(
    "Predictive analytics project using daily HHS UAC reporting data"
)


# ============================================================
# FILE PATHS
# ============================================================

ARTIFACT_DIR = "artifacts"

DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "daily_data.csv"
)

CARE_MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "care_load_rf.pkl"
)

DISCHARGE_MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "discharge_rf.pkl"
)

TRANSFER_MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "transfer_rf.pkl"
)

RESIDUAL_PATH = os.path.join(
    ARTIFACT_DIR,
    "validation_residuals.pkl"
)

COMPARISON_PATH = os.path.join(
    ARTIFACT_DIR,
    "model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    data = data.set_index("Date")

    return data


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    care_model = joblib.load(
        CARE_MODEL_PATH
    )

    discharge_model = joblib.load(
        DISCHARGE_MODEL_PATH
    )

    transfer_model = joblib.load(
        TRANSFER_MODEL_PATH
    )

    return {
        "care_load": care_model,
        "discharge": discharge_model,
        "transfer": transfer_model
    }


# ============================================================
# LOAD OTHER ARTIFACTS
# ============================================================

@st.cache_data
def load_residuals():

    return joblib.load(
        RESIDUAL_PATH
    )


@st.cache_data
def load_comparison():

    return pd.read_csv(
        COMPARISON_PATH
    )


daily = load_data()

models = load_models()

residuals = load_residuals()

comparison = load_comparison()


# ============================================================
# TARGETS
# ============================================================

TARGETS = {

    "care_load":
        "Children in HHS Care",

    "discharge":
        "Children discharged from HHS Care",

    "transfer":
        "Children transferred out of CBP custody"
}


# ============================================================
# FEATURE ENGINEERING FOR FUTURE PREDICTION
# ============================================================

def make_feature_row(history, date):

    s = pd.Series(
        history,
        index=pd.to_datetime(
            history.index
        )
    ).sort_index()

    row = {}

    # Lag features
    for lag in [
        1,
        2,
        3,
        7,
        14,
        21,
        28
    ]:

        row[f"lag_{lag}"] = float(
            s.iloc[-lag]
        )

    # Rolling statistics
    for window in [
        7,
        14,
        28
    ]:

        values = s.iloc[-window:]

        row[
            f"roll_mean_{window}"
        ] = float(
            values.mean()
        )

        row[
            f"roll_std_{window}"
        ] = float(
            values.std()
        ) if len(values) > 1 else 0.0

    # Calendar features

    row["day_of_week"] = (
        date.dayofweek
    )

    row["month"] = (
        date.month
    )

    row["day_of_year"] = (
        date.dayofyear
    )

    row["week_of_year"] = int(
        date.isocalendar().week
    )

    return pd.DataFrame(
        [row],
        index=[date]
    )


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    model,
    history,
    horizon
):

    history = history.copy()

    future = []

    for _ in range(horizon):

        next_date = (
            history.index[-1]
            +
            pd.Timedelta(days=1)
        )

        X_next = make_feature_row(
            history,
            next_date
        )

        prediction = float(
            model.predict(X_next)[0]
        )

        # Number of children cannot be negative
        prediction = max(
            0,
            prediction
        )

        future.append(
            (
                next_date,
                prediction
            )
        )

        history.loc[
            next_date
        ] = prediction

    return pd.Series(
        dict(future)
    )


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_series(
    key,
    horizon
):

    target = TARGETS[key]

    history = daily[target].copy()

    model = models[key]

    return recursive_forecast(
        model,
        history,
        horizon
    )


# ============================================================
# APPROXIMATE FORECAST RANGE
# ============================================================

def forecast_interval(
    key,
    forecast
):

    # Random Forest validation residuals
    values = np.asarray(
        residuals[key]["Random Forest"],
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        spread = max(
            1.0,
            float(
                daily[
                    TARGETS[key]
                ].std()
            )
        )

        lower = (
            forecast
            -
            1.645 * spread
        )

        upper = (
            forecast
            +
            1.645 * spread
        )

    else:

        lower_residual = np.quantile(
            values,
            0.05
        )

        upper_residual = np.quantile(
            values,
            0.95
        )

        lower = (
            forecast
            +
            lower_residual
        )

        upper = (
            forecast
            +
            upper_residual
        )

    return (
        lower.clip(lower=0),
        upper.clip(lower=0)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Forecast Settings"
)


horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [7, 14, 30],
    index=1
)


capacity = st.sidebar.number_input(
    "Care capacity threshold",
    min_value=1000,
    max_value=30000,
    value=10000,
    step=500
)


show_scenario = st.sidebar.checkbox(
    "Show high-pressure scenario",
    value=True
)


st.sidebar.markdown("---")


st.sidebar.write(
    "Last data date:",
    daily.index.max().strftime(
        "%d %b %Y"
    )
)


# ============================================================
# GENERATE FORECASTS
# ============================================================

care_forecast = forecast_series(
    "care_load",
    horizon
)

discharge_forecast = forecast_series(
    "discharge",
    horizon
)

transfer_forecast = forecast_series(
    "transfer",
    horizon
)


care_lower, care_upper = forecast_interval(
    "care_load",
    care_forecast
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

current_care = float(
    daily[
        "Children in HHS Care"
    ].iloc[-1]
)


peak_care = float(
    care_forecast.max()
)


average_discharge = float(
    discharge_forecast.mean()
)


capacity_days = int(
    (care_upper > capacity).sum()
)


# ============================================================
# KPI DISPLAY
# ============================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Current HHS Care Load",
    f"{current_care:,.0f}"
)


c2.metric(
    "Forecast Peak",
    f"{peak_care:,.0f}"
)


c3.metric(
    "Avg. Discharge Demand",
    f"{average_discharge:,.0f}"
)


c4.metric(
    "Days Above Capacity Range",
    f"{capacity_days}/{horizon}"
)


# ============================================================
# CARE LOAD FORECAST
# ============================================================

st.subheader(
    "1. Future Care Load Forecast"
)


fig = go.Figure()


# Last 90 days of history

history_plot = daily[
    "Children in HHS Care"
].tail(90)


fig.add_trace(
    go.Scatter(
        x=history_plot.index,
        y=history_plot.values,
        mode="lines",
        name="Historical Care Load"
    )
)


# Forecast

fig.add_trace(
    go.Scatter(
        x=care_forecast.index,
        y=care_forecast.values,
        mode="lines+markers",
        name="Random Forest Forecast"
    )
)


# Forecast range

fig.add_trace(
    go.Scatter(
        x=list(
            care_forecast.index
        )
        +
        list(
            care_forecast.index[::-1]
        ),

        y=list(
            care_upper.values
        )
        +
        list(
            care_lower.values[::-1]
        ),

        fill="toself",

        line=dict(
            color="rgba(0,0,0,0)"
        ),

        name="Approx. 90% Forecast Range"
    )
)


# Capacity line

fig.add_hline(
    y=capacity,
    line_dash="dash",
    annotation_text="Capacity threshold"
)


fig.update_layout(
    height=480,
    xaxis_title="Date",
    yaxis_title="Children",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.info(
    "The forecast range is an approximate 90% "
    "prediction range based on validation residuals. "
    "It is an analytical estimate rather than an "
    "official operational capacity probability."
)


# ============================================================
# DISCHARGE FORECAST
# ============================================================

st.subheader(
    "2. Discharge / Placement Demand"
)


fig2 = go.Figure()


fig2.add_trace(
    go.Bar(
        x=discharge_forecast.index,
        y=discharge_forecast.values,
        name="Predicted Discharges"
    )
)


fig2.update_layout(
    height=380,
    xaxis_title="Date",
    yaxis_title="Children discharged"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# FLOW PRESSURE
# ============================================================

st.subheader(
    "3. Intake / Exit Pressure"
)


pressure = (
    transfer_forecast
    -
    discharge_forecast
)


flow_df = pd.DataFrame({

    "Transfers from CBP":
        transfer_forecast,

    "Discharges from HHS":
        discharge_forecast,

    "Net pressure":
        pressure
})


st.line_chart(
    flow_df
)


st.caption(
    "Net pressure = predicted transfers "
    "from CBP − predicted HHS discharges."
)


# ============================================================
# HIGH PRESSURE SCENARIO
# ============================================================

if show_scenario:

    st.subheader(
        "4. Simple High-Pressure Scenario"
    )

    # Increase predicted transfers by 10%

    scenario_transfer = (
        transfer_forecast * 1.10
    )

    scenario_pressure = (
        scenario_transfer
        -
        discharge_forecast
    )

    scenario_care = (
        care_forecast.copy()
    )

    running_adjustment = 0.0

    for i, value in enumerate(
        scenario_pressure
    ):

        baseline = pressure.iloc[i]

        running_adjustment += max(
            0,
            float(
                value - baseline
            )
        )

        scenario_care.iloc[i] = (
            scenario_care.iloc[i]
            +
            running_adjustment
        )


    scenario_df = pd.DataFrame({

        "Baseline forecast":
            care_forecast,

        "High-pressure scenario":
            scenario_care
    })


    st.line_chart(
        scenario_df
    )


    st.caption(
        "Illustrative scenario: predicted "
        "transfers are increased by 10%. "
        "It is not an official HHS scenario."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader(
    "5. Model Comparison"
)


metric_target = st.selectbox(
    "Select target",
    [
        "care_load",
        "discharge",
        "transfer"
    ]
)


prefix = (
    metric_target
    +
    " - "
)


model_table = comparison[
    comparison["Model"]
    .str.startswith(prefix)
].copy()


model_table["Model"] = (
    model_table["Model"]
    .str.replace(
        prefix,
        "",
        regex=False
    )
)


st.dataframe(
    model_table.round(2),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Evaluation uses a chronological "
    "80/20 holdout. Lower MAE, RMSE and "
    "MAPE indicate smaller historical "
    "validation error."
)


# ============================================================
# RECENT DATA
# ============================================================

with st.expander(
    "View recent data"
):

    st.dataframe(
        daily.tail(30).round(2),
        use_container_width=True
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Project methodology"
):

    st.markdown(
        """
### Data Preparation

- Removed completely blank rows.
- Converted Date to datetime.
- Converted numerical columns to numeric values.
- Created a continuous daily time series.
- Interpolated missing reporting dates.

### Feature Engineering

- Lag values: 1, 2, 3, 7, 14, 21 and 28 days.
- 7, 14 and 28-day rolling mean.
- 7, 14 and 28-day rolling standard deviation.
- Day of week.
- Month.
- Day of year.
- Week of year.

### Models

- Naive persistence baseline.
- Random Forest regression.
- SARIMA.

### Evaluation

- Chronological 80/20 split.
- MAE.
- RMSE.
- MAPE.

### Production Forecast

The Streamlit dashboard uses the Random Forest
models trained on the complete historical dataset.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student Project • Predictive Forecasting "
    "of Care Load & Placement Demand"
)
