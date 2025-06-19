#********************* Library importation
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import functions as fn
import datetime 
from datetime import date,timedelta,datetime

#********************** Page configuration
st.set_page_config(page_title="Covid Prediction", layout="wide")

model=None

# ************************** Functions
@st.cache_resource
def get_covid_cases():
    daily_covid_data=fn.covid_cases_data()
    model_input=fn.feature_compute(daily_covid_data)
    return daily_covid_data,model_input

def get_xgb_model():
     return fn.load_model()

def model_pred(model_input,model):
    pred_input=model_input.iloc[-1:,:]
    return fn.regression(pred_input,model)

daily_covid_data,model_input = get_covid_cases()
latest_date=daily_covid_data.Date.max()
predict_date=latest_date+timedelta(7)
model=get_xgb_model()
prediction=model_pred(model_input,model)

#*************************************************************************

#**************************** Streamlit 


st.markdown("## 🧬 COVID-19 Cases Forecasting App")

# Context section
st.markdown("""
            ***
    This project is an XGB model that predicts the aggregated COVID-19 positive cases in a 7 day window in England after a last available record
    The data is obtained via public API https://api.ukhsa-dashboard.data.gov.uk
    - **Input**:  Updated COVID-19 positive cases data from API
    - **Output**: Aggregated COVID-19 Cases forcast for a 7 day window from last available covid information in the API               
    
    **Disclaimers**: 
    - The implemented model was trained using data from 2024-04-01 to 2025-06-04.            
    - Covid data updates from the API has some days of lag versus current date (Sorry, don't blame it on me 😄 )                      
    - The model is re   trained monthly using Github Actions
            """
    )
st.markdown("---")

#***************************************** Dashboard section **************************************************

# Fancy prediction box
st.markdown(f"""
<div style='text-align: center; padding: 10px; border: 1.5px solid #4CAF50; border-radius: 10px; background-color: #F9F9F9;'>
    <h4 style='color: #333; margin-bottom: 10px;'>Predicted aggregated COVID-19 cases from {latest_date.strftime("%Y-%m-%d")} to {predict_date.strftime("%Y-%m-%d")}:</h4>
    <h2 style='color: #4CAF50; margin: 0;'>{int(prediction):,}</h2>
    <p style='color: #777; font-size: 0.9em;'>Based on data up to <strong>{latest_date.strftime("%Y-%m-%d")}</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ------------------ Row 2: Two Column Layout ------------------ #
col1, col2, col3 = st.columns([3,1,6])

# ---- Column 1: Daily Evolution Plot ---- #
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily_covid_data["Date"],
        y=daily_covid_data["Cases"],
        mode="lines",
        name="Daily Cases"
    ))
    fig1.update_layout(
        title="📅 Daily COVID-19 Case Evolution",
        xaxis_title="Date",
        yaxis_title="Cases"
    )
    st.plotly_chart(fig1, use_container_width=True)

# ---- Column 2: Aggregated Weekly + Prediction Subplot ---- #
with col3:
    fig2 = make_subplots(rows=1, cols=2, shared_xaxes=False,
                         vertical_spacing=0.15,
                         subplot_titles=("Weekly Aggregated Cases", "Next Week Prediction"))

    # Subplot 1: Line plot (history)
    fig2.add_trace(go.Scatter(
        x=model_input["Date"],
        y=model_input["Cases_Agg"],
        mode="lines",
        name="Weekly Aggregated"
    ), row=1, col=1)

    # Subplot 2: Bar plot (prediction)
    fig2.add_trace(go.Bar(
         x=["Next value"],
        y=[int(prediction)],
        name="Forecast",
        marker_color="#19899e"
    ), row=1, col=2)

    # Optional: Force a consistent Y-axis range
    fig2.update_yaxes(range=[0, model_input["Cases_Agg"].max() * 1.1], row=1, col=1)
    fig2.update_yaxes(range=[0, model_input["Cases_Agg"].max() * 1.1], row=1, col=2)

    fig2.update_layout(showlegend=False,
                       title_text="📊 Weekly Aggregated Cases & Forecast")

    st.plotly_chart(fig2, use_container_width=True)


#***************************************** Prediction section **************************************************

st.markdown("## 🧪 Try the Model with Your Own Input")

st.markdown("Enter the number of positive COVID-19 cases for the following days (lags). Consider **t** as today:")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    lag_3 = st.number_input("Cases at t-3", min_value=0, step=1)

with col2:
    lag_4 = st.number_input("Cases at t-4", min_value=0, step=1)

with col3:
    lag_5 = st.number_input("Cases at t-5", min_value=0, step=1)

with col4:
    lag_6 = st.number_input("Cases at t-6", min_value=0, step=1)

with col5:
    lag_7 = st.number_input("Cases at t-7", min_value=0, step=1)

with col6:
    lag_8 = st.number_input("Cases at t-8", min_value=0, step=1)


if st.button("Predict Based on Input"):
    # Format into DataFrame (must match training feature order!)
    user_input_df = pd.DataFrame([{
        "Cases_lag3": lag_3,
        "Cases_lag4": lag_4,
        "Cases_lag5": lag_5,
        "Cases_lag6": lag_6,
        "Cases_lag7": lag_7,
        "Cases_lag8": lag_8
    }])
    user_pred=model_pred(user_input_df,model)
    # Show result
    st.success(f"Predicted aggregated COVID-19 cases for next 7 days: **{int(user_pred):,}** aggregated cases")
