#********************* Library importation
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import functions as fn

#********************** Page configuration
st.set_page_config(page_title="Image Classifier", layout="centered")

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
    pred_input=model_input.iloc[:-1,:]
    return fn.regression(pred_input,model)

daily_covid_data,model_input = get_covid_cases()
latest_date=daily_covid_data.Date.max()
model=get_xgb_model()
prediction=model_pred(model_input,model)

#*************************************************************************

#**************************** Streamlit 


st.title("💉 COVID-19 Cases predictor 💉")

# Context section
st.markdown("""
            ***
    This project is an XGB model that predicts the aggregated COVID-19 positive cases in a 7 day window after a last available record
    The data is obtained via public API https://api.ukhsa-dashboard.data.gov.uk
    - **Input**:  Updated COVID-19 positive cases data from API
    - **Output**: Aggregated COVID-19 Cases forcast for a 7 day window from last available covid information in the API               
    
    - **Disclaimers**: The implemented model is pretrained using data from 2024-04-01 to 2025-06-04.
            ***
            """
    )

# Fancy prediction box
st.markdown(f"""
<div style='text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 15px; background-color: #F9F9F9;'>
    <h2 style='color: #333;'>Predicted aggregated COVID-19 cases for next 7 days:</h2>
    <h1 style='color: #4CAF50;'>{int(Z):,}</h1>
    <p style='color: #777;'>Based on data up to <strong>{latest_date}</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ------------------ Row 2: Two Column Layout ------------------ #
col1, col2 = st.columns(2)

# ---- Column 1: Daily Evolution Plot ---- #
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily_covid_data["Date"],
        y=daily_covid_data["Cases"],
        mode="lines+markers",
        name="Daily Cases"
    ))
    fig1.update_layout(
        title="📅 Daily COVID-19 Case Evolution",
        xaxis_title="Date",
        yaxis_title="Cases"
    )
    st.plotly_chart(fig1, use_container_width=True)

# ---- Column 2: Aggregated Weekly + Prediction Subplot ---- #
with col2:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=False,
                         vertical_spacing=0.15,
                         subplot_titles=("Weekly Aggregated Cases", "Next Week Prediction"))

    # Subplot 1: Line plot (history)
    fig2.add_trace(go.Scatter(
        x=model_input["Date"],
        y=model_input["Cases_Agg"],
        mode="lines+markers",
        name="Weekly Aggregated"
    ), row=1, col=1)

    # Subplot 2: Bar plot (prediction)
    fig2.add_trace(go.Bar(
        x=["Next Week"],
        y=[prediction],
        name="Forecast",
        marker_color="#FF4136"
    ), row=2, col=1)

    fig2.update_layout(height=600, showlegend=False,
                       title_text="📊 Weekly Aggregated Cases & Forecast")

    st.plotly_chart(fig2, use_container_width=True)


