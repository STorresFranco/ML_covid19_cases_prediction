import pandas as pd
import numpy as np
import requests
import datetime 
from datetime import date,timedelta
import xgboost
from xgboost import XGBRegressor
import scipy
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import io

def data_cases_update(h_covid_data):
    '''
    Updates historical COVID-19 data by appending new rows from latest CSV hosted on GitHub.
    '''
    # Download new CSV from GitHub
    url = "https://raw.githubusercontent.com/STorresFranco/ML_covid19_cases_prediction/main/artifacts/ukhsa-chart-download.csv"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            df_new = pd.read_csv(io.StringIO(response.text))
            df_new["Date"] = pd.to_datetime(df_new["date"], format="ISO8601")
            # Filter new dates not in h_covid_data
            new_rows = df_new[~df_new["Date"].isin(h_covid_data["Date"])]
            if new_rows.empty:
                print("We are up to date: No new records to add.")
            else:
                print(f"New data: {len(new_rows)} new records added.")
                h_covid_data = pd.concat([
                    h_covid_data,
                    pd.DataFrame({
                        "Year": new_rows["Date"].dt.year,
                        "Month": new_rows["Date"].dt.month,
                        "Day": new_rows["Date"].dt.day,
                        "Date": new_rows["Date"],
                        "Cases": new_rows["metric_value"]
                    })
                ], ignore_index=True)

                h_covid_data.drop_duplicates(subset="Date", inplace=True)
                h_covid_data.sort_values("Date", inplace=True)
                h_covid_data.reset_index(drop=True, inplace=True)
        except Exception as e:
            print("Failed response: Failed to parse CSV from response:", str(e))
    else:
        print("Failed fetch: Failed to fetch data. Status:", response.status_code)
    return h_covid_data

def data_concatenation():
    '''
    Description:
        Function used to read raw COVID-19 data and merge it with new data pulled from COVID-19 API
    Inputs:
        None
    Returns:
        h_covid_data (Dataframe): Dataframe used as input data to retrain the XGB model 
    '''
    # Reading raw data
    url="https://raw.githubusercontent.com/STorresFranco/ML_covid19_cases_prediction/main/artifacts/ukhsa-chart-download.csv" #Path to covid data file on Github
    raw_covid_data=pd.read_csv(url)

    #Creating dataframe to store results
    h_covid_data=pd.DataFrame({
        "Year":[],
        "Month":[],
        "Day":[],
        "Date":[],
        "Cases":[],
    })

    h_covid_data["Date"]=pd.to_datetime(raw_covid_data["date"],format="ISO8601")
    h_covid_data["Month"]=h_covid_data["Date"].dt.month
    h_covid_data["Year"]=h_covid_data["Date"].dt.year
    h_covid_data["Day"]=h_covid_data["Date"].dt.day
    h_covid_data["Cases"]=raw_covid_data["metric_value"]
    h_covid_data.sort_values(by="Date",axis="index",inplace=True)
    h_covid_data.reset_index(drop=True,inplace=True)

    # Updating the dataframe with the most recent info
    h_covid_data=data_cases_update(h_covid_data)

    return h_covid_data


def agg_lagged_cases(h_covid_data):
    ''' 
    Description:
        Function used to generate the required lagged covid features from t-3 to t-8 which are used as input data for the regression
        Also computes the aggregated cases on a 7 day rolling window
    Inputs:
        h_covid_data (Dataframe): Dataframe with historical information of COVID-19 Cases

    Returns
        h_covid_data (Dataframe): Updated dataframe with laaged cases features
    '''
    #Creating the aggregated feature
    h_covid_data["Cases_Agg"]=h_covid_data.Cases.rolling(window=7).sum()

    #Creating the lagged features
    for i in range(3,9): 
        h_covid_data[f"Cases_lag{i}"]=h_covid_data[["Cases"]].shift(i).values

    h_covid_data.dropna(inplace=True)

    return h_covid_data

def train_model(h_covid_data):
    '''
    Description:
        Function used to retrain the model over the available COVID-19 historical data

    Inputs: 
        h_covid_data (Dataframe):  Dataframe with historical information of COVID-19 Cases, laaged cases, and aggregated cases

    Returns
        model: Updated model after retrain and validation
    '''

    # ****************************** Test train split
    #Taking the train and test set
    train_data=h_covid_data.loc[:int(0.8*len(h_covid_data)),:]
    train_data=train_data[["Cases_lag3","Cases_lag4","Cases_lag5","Cases_lag6","Cases_lag7","Cases_lag8","Cases_Agg"]]

    test_data=h_covid_data.loc[int(0.8*len(h_covid_data))+1:,:]
    test_data=test_data[["Cases_lag3","Cases_lag4","Cases_lag5","Cases_lag6","Cases_lag7","Cases_lag8","Cases_Agg"]]

    # ****************************** Train
    #Split X and y variables
    X_train=train_data.drop("Cases_Agg",axis="columns")
    y_train=train_data.Cases_Agg
    y_train_log=np.log1p(y_train)

    X_test=test_data.drop("Cases_Agg",axis="columns")
    y_test=test_data.Cases_Agg

    #Scale input
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    joblib.dump(scaler, 'streamlit/scaler.gz') #Dumping the scaler after data update

    #Fit the model
    xgb_model=XGBRegressor()
    xgb_model.fit(X_train_scaled,y_train_log)

    # ****************************** Evaluation
    #Train evaluation
    train_pred=xgb_model.predict(X_train_scaled)
    train_pred=np.expm1(train_pred)
    train_rmse=root_mean_squared_error(y_train,train_pred)
    r2_train=r2_score(y_train,train_pred)

    #Test evaluation
    test_pred=xgb_model.predict(X_test_scaled)
    test_pred=np.expm1(test_pred)
    test_rmse=root_mean_squared_error(y_test,test_pred)
    r2_test=r2_score(y_test,test_pred)

    if r2_test < 0.60: #Case the model performance drop below an output 
        raise ValueError(f"Warning!!!: R² too low: {r2_test:.3f}. Model not saved.")
    else:
        xgb_model.save_model('streamlit/xgb_final_model.json') #Saving the final model


    return train_rmse,r2_train,test_rmse,r2_test,xgb_model


#*************************************************************************
if __name__=="__main__":
    h_covid_data=data_concatenation()
    model_input=agg_lagged_cases(h_covid_data)
    train_rmse,r2_train,test_rmse,r2_test,xgb_model=train_model(model_input)

    print(f"Succes!!: Retraining complete")
    print(f"Train RMSE: {train_rmse:.2f} | R²: {r2_train:.3f}")
    print(f"Test  RMSE: {test_rmse:.2f} | R²: {r2_test:.3f}")

