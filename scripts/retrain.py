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

def data_cases_update(h_covid_data):
    '''
    Description:
        Function that updates historical positive cases (h_covid_data) using an API request from source data and if the date is not duplicated
    Inputs
        h_covid_data (Dataframe): Contains historical COVID-19 data that is to be updated    
    Outputs:
        h_covid_data (Dataframe): Updated dataframe
    '''
    curr_date=date.today()
    day_data=curr_date.day      #Check current day for data update
    month_data=curr_date.month
    year_data=curr_date.year
    new_record=True             #Control variable to check for data existence

    while new_record:

        #Request to data source
        url = "https://raw.githubusercontent.com/STorresFranco/ML_covid19_cases_prediction/main/artifacts/ukhsa-chart-download.csv"
        response=requests.get(url)

        #Checking for validity on request
        if response.status_code==200: #Data importation succesful
           try:
                data = response.json()
                print("Covid data request successful with status", response.status_code)
            except ValueError:
                print("❌ Response was not JSON! Here's what we got:")
                print(response.text)
                break
        else:
            print("Covid data request failed with status ",response.status_code)
            break
        #checking for data existnce
        if len(data["results"])>0: #Case data exist take new values          
            curr_date=pd.Timestamp(curr_date)          
  
            if curr_date in h_covid_data.Date.values: #Validate if data is up to date. In case it is end
                 h_covid_data.sort_values(by="Date",axis="index",inplace=True) #Ensure data is sorted
                 h_covid_data.reset_index(drop=True,inplace=True)
                 print(f"Historical data range for Covid-19 positive cases from {h_covid_data.Date.min()} to {h_covid_data.Date.max()}")
                 break                              
  
            else:                                   
                print(f"New record added to covid data for date {curr_date}")
                results=data["results"][0]
                new_data=pd.DataFrame({     #Dataframe to concat new info
                        "Year":[pd.to_datetime(results["date"]).year],
                        "Month":[pd.to_datetime(results["date"]).month],
                        "Day":[pd.to_datetime(results["date"]).day],
                        "Date":[pd.to_datetime(results["date"],format="ISO8601")],
                        "Cases":[results["metric_value"]],
                    })
                h_covid_data=pd.concat([h_covid_data,new_data],axis="index",ignore_index=True)
 
                h_covid_data.Date=pd.to_datetime(h_covid_data.Date) #Ensuring format consistency
                h_covid_data.drop_duplicates(ignore_index=True,inplace=True)
                
                curr_date=curr_date-timedelta(1) #Move to next date 
                day_data=curr_date.day           
                month_data=curr_date.month
                year_data=curr_date.year                

        else:
                curr_date=curr_date-timedelta(1) #Take a previous date
                day_data=curr_date.day           #Check current day for data update
                month_data=curr_date.month
                year_data=curr_date.year


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

    X_test=test_data.drop("Cases_Agg",axis="columns")
    y_test=test_data.Cases_Agg
    y_test_log=np.log1p(y_test)

    #Scale input
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    joblib.dump(scaler, 'streamlit/scaler.gz') #Dumping the scaler after data update

    #Fit the model
    xgb_model=XGBRegressor()
    xgb_model.fit(X_train_scaled,y_test_log)

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
        raise ValueError(f"🚨 R² too low: {r2_test:.3f}. Model not saved.")
    else:
        xgb_model.save_model('streamlit/xgb_final_model.json') #Saving the final model


    return train_rmse,r2_train,test_rmse,r2_test,xgb_model


#*************************************************************************
if __name__=="__main__":
    h_covid_data=data_concatenation()
    model_input=agg_lagged_cases(h_covid_data)
    train_rmse,r2_train,test_rmse,r2_test,xgb_model=train_model(model_input)

    print(f"✅ Retraining complete")
    print(f"Train RMSE: {train_rmse:.2f} | R²: {r2_train:.3f}")
    print(f"Test  RMSE: {test_rmse:.2f} | R²: {r2_test:.3f}")


