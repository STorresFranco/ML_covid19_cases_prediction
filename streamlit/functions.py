import pandas as pd
import numpy as np
import requests
import datetime 
from datetime import date,timedelta
import xgboost
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib


def covid_cases_data():
    '''
    Description:
        Function tp update historical positive COVID-19 cases (h_covid_data) using an API request from source data and if the date is not duplicated

    Inputs:
        None    

    Returns:
        h_covid_data (Dataframe): Updated dataframe with availabke positive COVID-19 cases

    Notes:
        The function reads the github file historical_covid_data.csv as starting point for the update        
    '''
   
   #*************** Creating and formating the dataframe of historical data 
    h_covid_data=pd.DataFrame({
        "Year":[],
        "Month":[],
        "Day":[],
        "Date":[],
        "Cases":[],
    })
    url="https://raw.githubusercontent.com/STorresFranco/ML_covid19_cases_prediction/main/artifacts/ukhsa-chart-download.csv" #Path to covid data file on Github
    raw_covid_data=pd.read_csv(url)

    h_covid_data["Date"]=pd.to_datetime(raw_covid_data["date"],format="ISO8601")
    h_covid_data["Month"]=h_covid_data["Date"].dt.month
    h_covid_data["Year"]=h_covid_data["Date"].dt.year
    h_covid_data["Day"]=h_covid_data["Date"].dt.day
    h_covid_data["Cases"]=raw_covid_data["metric_value"]
    h_covid_data.sort_values(by="Date",axis="index",inplace=True)
    h_covid_data.reset_index(drop=True,inplace=True)
   
    #************** Pulling data from the API
    curr_date=date.today()
    day_data=curr_date.day      #Check current day for data update
    month_data=curr_date.month
    year_data=curr_date.year
    new_record=True             #Control variable to check for data existence

    while new_record:

        #Request to data source
        url=f"https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/COVID-19/geography_types/Nation/geographies/England/metrics/COVID-19_cases_casesByDay?date={year_data}-{month_data}-{day_data}"
        response=requests.get(url)

        #Checking for validity on request
        if response.status_code==200: #Data importation succesful
            data=response.json()
            print("Covid data request succesful with status", response.status_code)
        else:
            print("Covid data request failed with status ",response.status_code)

        #checking for data existnce
        if len(data["results"])>0: #Case data exist take new values          
            curr_date=pd.Timestamp(curr_date)          
  
            if curr_date in h_covid_data.Date.values: #Validate if data is up to date. In case it is end
                 h_covid_data.sort_values(by="Date",axis="index",inplace=True) #Ensure data is sorted
                 h_covid_data.reset_index(drop=True,inplace=True)
                 print(f"Historical data range for Covid-19 positive cases from {h_covid_data.Date.min()} to {h_covid_data.Date.max()}")
                 h_covid_data.to_csv("historical_covid_data.csv")
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

def feature_compute(h_covid_data):
    '''
    Description
        Function to compute the input data from daily COVID-19 cases required by the regressor to forecast aggregated cases over next 7 days.

    Inputs
        h_covid_data (Dataframe): Dataframe with daily COVID-19 positive cases

    Returns
        h_covid_data (Dataframe): Updated dataframe with the following features
            * Cases_lag3: Positive cases at t-3 from last available date data
            * Cases_lag5: Positive cases at t-5 from last available date data
            * Cases_lag6: Positive cases at t-6 from last available date data
            * Cases_lag7: Positive cases at t-7 from last available date data
         
    '''
    model_input=h_covid_data.copy()

    # Aggregated data for covid
    model_input["Cases_Agg"]=model_input.Cases.rolling(window=7).sum()

    # Lagged data for covid
    for i in range(3,9): 
        model_input[f"Cases_lag{i}"]=model_input[["Cases"]].shift(i).values
    
    model_input.dropna(inplace=True)
    return model_input

def load_model():
    '''
    Description
        Function to load a pretrained model for prediction purposes

    Inputs
        None
    
    Returns
        model: loaded model

    '''
    #Loading the model
    xgb_model_path= f"xgb_final_model.json"
    xgb_model = XGBRegressor()
    xgb_model.load_model(xgb_model_path)
    return xgb_model

def regression(model_input,xgb_model):
    '''
    Description
        Function to predict the accumulated positive COVID-19 cases over the next 7 days

    Inputs
        h_covid_data (Dataframe): Dataframe with daily COVID-19 positive cases and derived lagged features

    Returns
        prediction

    '''
  

    #Loading the scaler
    scaler=joblib.load("scaler.gz")

    #Scaling the data
    X_data=model_input[["Cases_lag3","Cases_lag4","Cases_lag5","Cases_lag6","Cases_lag7","Cases_lag8"]]
    X_data=scaler.transform(X_data)

    #Prediction for next 7 days
    prediction=xgb_model.predict(X_data)
    prediction=np.expm1(prediction)

    return prediction
