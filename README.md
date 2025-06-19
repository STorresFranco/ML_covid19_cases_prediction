# 🦠 COVID-19 Cases Forecast in England

**Project Status:** ✅ Deployed | 🔁 Auto-Retraining via GitHub Actions

In this project, I develop a model capable of forecasting aggregated COVID-19 cases over a 7-day window using real data from public APIs. The data includes daily positive COVID-19 case counts in England and meteorological information from a specific location in the country. Throughout the project, I explored the potential relationship between COVID-19 cases and meteorological variables to improve model performance. The acceptance criterion for the model was an R² score above 0.7.

---

## 📊 Prototype

The prototyping phase involved collecting one year of historical daily data (from April 2024 to May 2025) from public sources:

* **COVID-19 Data:** Positive case counts
* **Weather Data:** Minimum, mean, and maximum temperature, accumulated precipitation, snowfall, and max wind speed

The following steps were carried out:

* Data cleaning and statistical exploration, including stationarity testing
* Feature engineering to derive meaningful explanatory variables:

  * Lagged COVID-19 case values (t-1 to t-8)
  * Temperature range and rolling means
  * Fourier components to model seasonality
* Feature selection based on correlation and cointegration with the target
* Model selection using walk-forward validation and Bayesian optimization via Optuna (XGBoost vs. LightGBM)
* Model tuning based on feature importance analysis

---

## 🔧 Modeling Conclusions

* ✅ **XGBoost** was selected as the final model due to superior performance
* ❌ No significant correlation was found between weather data and case counts
* 🔹 The most relevant features were lagged COVID-19 cases from t-3 to t-8
* ⭐ Achieved R² Score: **0.77**

---

## 🔄 How the Model Works

* Daily case data is retrieved from a GitHub-hosted CSV
* Lag and rolling features are dynamically calculated
* Model retrains automatically only if R² ≥ 0.70
* Updated model is served via Streamlit frontend

---

## 💡 Model Features

* **📊 Model:** XGBoost Regressor trained on lagged COVID-19 case data
* **🔁 CI/CD:** Automated retraining via GitHub Actions
* **🌍 Live Data:** Integrated with [UKHSA](https://coronavirus.data.gov.uk/)
* **📉 Engineered Features:**

  * 7-day rolling case sums
  * Lag variables: t-3 to t-8
* **📦 Deployment:** Publicly accessible Streamlit app

---

## 🌟 Highlights

One of the most compelling aspects of this project is the **hyperparameter tuning using Optuna** in combination with **walk-forward validation**, mimicking cross-validation in a time-series context.

R² was chosen over RMSE/MSE to maintain performance balance between case spikes and troughs. RMSE/MSE would bias the model toward minimizing errors during peaks only.

---

## 📂 Repository Structure

```
.github/workflows/       -> GitHub Actions pipeline for retraining
prototype/               -> Jupyter Notebook for initial analysis and modeling
scripts/                 -> Python retraining logic used by the CI/CD pipeline
streamlit/               -> Live app interface and model I/O files
```

---

## 📃 Data Sources and Credits

**COVID-19 Case Data:**

* [https://ukhsa-dashboard.data.gov.uk/](https://ukhsa-dashboard.data.gov.uk/)
* [https://api.ukhsa-dashboard.data.gov.uk](https://api.ukhsa-dashboard.data.gov.uk)

**Weather Data:**

* Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
* Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023). ERA5 hourly data on single levels from 1940 to present [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
* Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC
* Schimanke S., Ridal M., Le Moigne P., Berggren L., Undén P., Randriamampianina R., Andrea U., Bazile E., Bertelsen A., Brousseau P., Dahlgren P., Edvinsson L., El Said A., Glinton M., Hopsch S., Isaksson L., Mladek R., Olsson E., Verrelle A., Wang Z.Q. (2021). CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.622A565A
