import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 UAC Care Load & Placement Demand Forecast")
st.caption("Predictive analytics project using daily HHS UAC reporting data")

ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "daily_data.csv")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "models.pkl")
RESIDUAL_PATH = os.path.join(ARTIFACT_DIR, "validation_residuals.pkl")
COMPARISON_PATH = os.path.join(ARTIFACT_DIR, "model_comparison.csv")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    return data

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_residuals():
    return joblib.load(RESIDUAL_PATH)

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

daily = load_data()
models = load_models()
residuals = load_residuals()
comparison = load_comparison()

TARGETS = {
    "care_load": "Children in HHS Care",
    "discharge": "Children discharged from HHS Care",
    "transfer": "Children transferred out of CBP custody"
}

def make_feature_row(history, date):
    s = pd.Series(history, index=pd.to_datetime(history.index)).sort_index()
    row = {}

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        row[f"lag_{lag}"] = float(s.iloc[-lag])

    for window in [7, 14, 28]:
        vals = s.iloc[-window:]
        row[f"roll_mean_{window}"] = float(vals.mean())
        row[f"roll_std_{window}"] = float(vals.std()) if len(vals) > 1 else 0.0

    row["day_of_week"] = date.dayofweek
    row["month"] = date.month
    row["day_of_year"] = date.dayofyear
    row["week_of_year"] = int(date.isocalendar().week)

    return pd.DataFrame([row], index=[date])

def recursive_rf_forecast(model, history, horizon):
    history = history.copy()
    future = []

    for _ in range(horizon):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        x_next = make_feature_row(history, next_date)
        pred = float(model.predict(x_next)[0])
        pred = max(0, pred)
        future.append((next_date, pred))
        history.loc[next_date] = pred

    return pd.Series(dict(future))

def forecast_series(key, model_name, horizon):
    target = TARGETS[key]
    history = daily[target].copy()

    if model_name == "Random Forest":
        return recursive_rf_forecast(models[key]["Random Forest"], history, horizon)

    if model_name == "Naive":
        next_dates = pd.date_range(
            history.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )
        return pd.Series(history.iloc[-1], index=next_dates)

    # SARIMA
    result = models[key]["SARIMA"]
    return result.get_forecast(steps=horizon).predicted_mean.clip(lower=0)

def interval_for(key, model_name, forecast):
    vals = np.asarray(residuals[key][model_name], dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        spread = max(1.0, float(daily[TARGETS[key]].std()))
        lower = forecast - 1.645 * spread
        upper = forecast + 1.645 * spread
    else:
        low_resid = np.quantile(vals, 0.05)
        high_resid = np.quantile(vals, 0.95)
        lower = forecast + low_resid
        upper = forecast + high_resid

    return lower.clip(lower=0), upper.clip(lower=0)

# ---------------- Sidebar ----------------
st.sidebar.header("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [7, 14, 30],
    index=1
)

model_name = st.sidebar.selectbox(
    "Forecast model",
    ["Random Forest", "Naive", "SARIMA"]
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
st.sidebar.write("Last data date:", daily.index.max().strftime("%d %b %Y"))

# ---------------- KPI cards ----------------
care_forecast = forecast_series("care_load", model_name, horizon)
discharge_forecast = forecast_series("discharge", model_name, horizon)
transfer_forecast = forecast_series("transfer", model_name, horizon)

care_lower, care_upper = interval_for(
    "care_load", model_name, care_forecast
)

recent_care = float(daily["Children in HHS Care"].iloc[-1])
avg_discharge = float(discharge_forecast.mean())
peak_care = float(care_forecast.max())
breach_days = int((care_upper > capacity).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current HHS Care Load", f"{recent_care:,.0f}")
c2.metric("Forecast Peak", f"{peak_care:,.0f}")
c3.metric("Avg. Discharge Demand", f"{avg_discharge:,.0f}")
c4.metric("Days Above Capacity Range", f"{breach_days}/{horizon}")

# ---------------- Main care-load chart ----------------
st.subheader("1. Future Care Load Forecast")

fig = go.Figure()

history_plot = daily["Children in HHS Care"].tail(90)
fig.add_trace(go.Scatter(
    x=history_plot.index,
    y=history_plot.values,
    mode="lines",
    name="Historical Care Load"
))

fig.add_trace(go.Scatter(
    x=care_forecast.index,
    y=care_forecast.values,
    mode="lines+markers",
    name=f"{model_name} Forecast"
))

fig.add_trace(go.Scatter(
    x=list(care_forecast.index) + list(care_forecast.index[::-1]),
    y=list(care_upper.values) + list(care_lower.values[::-1]),
    fill="toself",
    line=dict(color="rgba(0,0,0,0)"),
    name="Approx. 90% Forecast Range"
))

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

st.plotly_chart(fig, use_container_width=True)

st.info(
    "The forecast range is an approximate 90% prediction range based on "
    "validation residuals. It should be treated as an analytical estimate, "
    "not an official operational capacity probability."
)

# ---------------- Discharge demand ----------------
st.subheader("2. Discharge / Placement Demand")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=discharge_forecast.index,
    y=discharge_forecast.values,
    name="Predicted Discharges"
))
fig2.update_layout(
    height=380,
    xaxis_title="Date",
    yaxis_title="Children discharged"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- Flow pressure ----------------
st.subheader("3. Intake / Exit Pressure")

pressure = transfer_forecast - discharge_forecast

flow_df = pd.DataFrame({
    "Transfers from CBP": transfer_forecast,
    "Discharges from HHS": discharge_forecast,
    "Net pressure": pressure
})

st.line_chart(flow_df)

st.caption(
    "Net pressure = predicted transfers from CBP − predicted HHS discharges. "
    "Positive values indicate more incoming flow than predicted exits."
)

# ---------------- Scenario ----------------
if show_scenario:
    st.subheader("4. Simple High-Pressure Scenario")

    # Illustrative scenario: 10% higher predicted transfers.
    scenario_transfer = transfer_forecast * 1.10
    scenario_pressure = scenario_transfer - discharge_forecast

    scenario_care = care_forecast.copy()
    running_adjustment = 0.0

    for i, value in enumerate(scenario_pressure):
        baseline_pressure = pressure.iloc[i]
        running_adjustment += max(0, float(value - baseline_pressure))
        scenario_care.iloc[i] = scenario_care.iloc[i] + running_adjustment

    scenario_df = pd.DataFrame({
        "Baseline forecast": care_forecast,
        "High-pressure scenario": scenario_care
    })

    st.line_chart(scenario_df)
    st.caption(
        "Illustrative scenario only: predicted transfers are increased by 10%. "
        "This is not a probability forecast and does not represent an official HHS scenario."
    )

# ---------------- Model comparison ----------------
st.subheader("5. Model Comparison")

metric_target = st.selectbox(
    "Select target for model comparison",
    ["care_load", "discharge", "transfer"]
)

prefix = metric_target + " - "
model_table = comparison[
    comparison["Model"].str.startswith(prefix)
].copy()

model_table["Model"] = model_table["Model"].str.replace(prefix, "", regex=False)

st.dataframe(
    model_table.sort_values("MAE").round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Evaluation uses a chronological 80/20 holdout. Lower MAE/RMSE/MAPE means "
    "smaller historical validation error; this table is descriptive rather than a guarantee of future performance."
)

# ---------------- Data overview ----------------
with st.expander("View recent data"):
    st.dataframe(daily.tail(30).round(2), use_container_width=True)

with st.expander("Project methodology"):
    st.markdown("""
    **Data preparation**
    - Removed completely blank rows.
    - Converted dates to datetime.
    - Converted comma-formatted numeric fields to numbers.
    - Created a daily time series and interpolated missing reporting dates.

    **Features**
    - Lag values: 1, 2, 3, 7, 14, 21 and 28 days.
    - 7, 14 and 28 day rolling mean and standard deviation.
    - Day of week, month, day of year and week of year.

    **Models**
    - Naive persistence baseline.
    - Random Forest regression.
    - SARIMA with weekly seasonality.

    **Validation**
    - Chronological 80/20 split.
    - No random shuffling.
    - MAE, RMSE and MAPE.
    """)

st.markdown("---")
st.caption("Student project dashboard • Predictive Forecasting of Care Load & Placement Demand")
