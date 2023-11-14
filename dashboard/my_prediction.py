import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os

def main():

    forecast = get_forecast('hel','pm10')
    folder = os.fsencode("data\clean_monthly_average_data")
    combined_data = import_and_combine_csv(folder)

    values=forecast_error(combined_data, 'hel', 'pm10')

    print(values)

def get_forecast(city,pollutant,data=None,model=None,periods=60,freq='M'):
    """
    Returns forecast as a dataframe

        Parameters:
            city (str) : the city the forecast is for
            pollutant (str) : the pollutant the forecast is for
        
        Optional Parameters:
            data (Pandas.DataFrame) : if data is not specified, will fetch the monthly data en create the neccisary data frame,
                specifying the data will speed up execution.
            model (Prophet) : see Prohet documentation at: https://facebook.github.io/prophet/docs/quick_start.html#python-api
            periods (int) : The number of periods to include in the prediction
            freq (str) : The frequency of the periods. Any valid frequency for pd.date_range, such as 'D' or 'M'. (default='M')

        Returns:
            forecast (Pandas.DataFrame) : dataframe of the forecast
                    For more, see the Prophet docuentation on: https://facebook.github.io/prophet/docs/quick_start.html#python-api

    """
    if data is None:
        folder = os.fsencode("data\clean_monthly_average_data")
        all_data = import_and_combine_csv(folder)
        data = _get_sub_df(all_data,city,pollutant)

    if model is None:
        model = get_forecast_model(city,pollutant,data)
    
    future = model.make_future_dataframe(periods=periods,freq=freq,include_history=False)
    forecast = model.predict(future)

    return forecast

def get_forecast_model(city,pollutant,data=None):
    """
    Returns a forecast model

        Parameters:
            city (str) : the city the model needs to be based on
            pollutant (str) : the pollutant the model needs to based on
        
        Optional Parameters:
            data (Pandas.DataFrame) : if data is not specified, will fetch the monthly data en create the neccisary data frame,
                specifying the data will speed up execution.

        Returns:
            model (Prophet) : see Prohet documentation at: https://facebook.github.io/prophet/docs/quick_start.html#python-api


    """
    if data is None:
        folder = os.fsencode("data\clean_monthly_average_data")
        all_data = import_and_combine_csv(folder)
        data = _get_sub_df(all_data,city,pollutant)

    model = Prophet()
    model.fit(data)

    return model

def forecast_pollutant_future(data, city, pollutant, target_year=None, target_month=None,make_plot=False):
    """
    Returns the forecast for the provided pollutant in the specified city

        Parameters:
            data (Pandas.DataFrame)
            city (str) : the city name as in the data column
            pollutant (str)

            Optional arguments:
                target_year (int) Default=None;
                target_month (int) Default=None : If a target year and month are provided, find the forecasted value for that period
                make_plot (Boolean) Default=False: if set to true, will plot the forecast

            Returns:
                forecast (Pandas.DataFrame) : dataframe of the forecast with the:
                    - date ['ds'],
                    - predicted value ['yhat'],
                    - lower bound of uncertainty interval ['yhat_lower']
                    - upper bound of uncertainty interval ['yhat_upper']
                For more, see the forecast docuentation on: https://facebook.github.io/prophet/docs/quick_start.html#python-api

    """
    # Filter data for the specified city and pollutant
    prophet_df = _get_sub_df(data,city,pollutant)

    # Initialize and fit the Prophet model
    m = Prophet()
    m.fit(prophet_df)

    # Create a future DataFrame for forecasting
    future = m.make_future_dataframe(periods=60, freq='M') 

    # Forecast
    forecast = m.predict(future)

    if make_plot:
        _plot_forecast(m,prophet_df,forecast,pollutant,city)

    # If a target year and month are provided, find the forecasted value for that period
    if target_year and target_month:
        forecasted_value_row = forecast[(forecast['ds'].dt.year == target_year) & (forecast['ds'].dt.month == target_month)]
        if not forecasted_value_row.empty:
            forecasted_value = forecasted_value_row['yhat'].iloc[0]
            print(f"The forecasted {pollutant} level for {city} in {target_month:02d}-{target_year} is {forecasted_value:.2f}")
        else:
            print(f"No forecast available for {target_month:02d}-{target_year} in {city}.")

    # Return the forecast DataFrame
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _plot_forecast(model,prophet_df,forecast,pollutant,city):
    # Plot the historical and forecasted data
    plt.figure(figsize=(10, 6))
    plt.plot(prophet_df['ds'], prophet_df['y'], label='Historical', color='blue')

    # We only plot the 'yhat' values for the forecasted dates, not the historical dates
    forecasted = forecast[forecast['ds'] > prophet_df['ds'].max()]
    plt.plot(forecasted['ds'], forecasted['yhat'], label='Forecast', color='red', linestyle='--')
    plt.plot(forecasted['ds'], forecasted['yhat_upper'], label='Forecast', color='orange', linestyle='--')
    plt.plot(forecasted['ds'], forecasted['yhat_lower'], label='Forecast', color='orange', linestyle='--')

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel(f'{pollutant} Level')
    plt.title(f'Historical and Forecasted {pollutant} Levels in {city}')
    plt.legend()
    plt.show()

    # Plot the forecast components
    fig = model.plot_components(forecast)
    plt.show()

def import_and_combine_csv(folder):
    """
    Returns single dataframe including data in specified folder

            Parameters:
                folder (bytes) : os.fsencoded path to the folder containing the data files as csv's

            Returns:
                combined_df (Pandas.DataFrame)
    """
    dataframes = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        file_str = os.fsdecode(file_path)
        df = pd.read_csv(file_str)
        #Extract city name from before the .csv 
        city_name = file_str.split('_')[-2].split('.')[0]  
        df['city'] = city_name
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def _calculate_error_rate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return mae, mse, rmse

def forecast_error(data, city, pollutant, test_size=20, make_plot=False):
    """
    Prints the errors of the forecast on the test_size latest datapoints for the model build for the specified city and pollutant

        Parameters:
            data (Pandas.DataFrame)
            city (str) : the city name as in the data column
            pollutant (str)
            
            Optional:
                test_size (int) Default=20: the number of the latest data points to calculated the error on
                make_plot (Boolean) Default=False: if set to true, will plot the forecast
        
        Returns: mae, mse, rmse, forecast
            mae (int) : mean absolute error
            mse (int) : mean squared error
            rmse (int) : root of mse
            forecast (Pandas.DataFrame) : the forecast built for error calculations
    """
    # Filter data for the specified city and pollutant
    prophet_df = _get_sub_df(data,city,pollutant)

    train_df, test_df = split_train_test(prophet_df,test_size)

    # Initialize and fit the Prophet model
    m = Prophet()
    m.fit(train_df)

    # Create a future DataFrame for forecasting
    future_periods = (test_df['ds'].max() - train_df['ds'].max()).days // 30 + 1
    future = m.make_future_dataframe(periods=future_periods, freq='M') 

    # Forecast
    forecast = m.predict(future)

    #Error calculation 
    mae, mse, rmse = _get_forecast_error()

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    if make_plot:
        # Plot the historical and forecasted data
        _plot_forecast(m,prophet_df,forecast,pollutant,city)

    return mae, mse, rmse, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def _get_forecast_error(forecast,test_df,test_size):
    forecasted_test = forecast[-test_size:]
    y_true = test_df['y'].values
    y_pred = forecasted_test['yhat'][-test_size:].values

    mae, mse, rmse = _calculate_error_rate(y_true,y_pred)

    return mae, mse, rmse

def _get_sub_df(data, city, pollutant):
    """Returns a data frame with the entries of data for the specified city and pollutant"""
    city_data = data[(data['city'] == city)]
    filtered_df = city_data[['date', pollutant]].dropna().rename(columns={'date': 'ds', pollutant: 'y'})
    filtered_df['ds'] = pd.to_datetime(filtered_df['ds'])
    filtered_df = filtered_df.set_index('ds').asfreq('M').reset_index()
    return filtered_df

def split_train_test(dataframe,test_size=20):
    """
    Returns two dataframes based on the dataframe passed, split on test_size

        Parameters:
            dataframe (Pandas.DataFrame): the original dataframe
            test_size (int): the size of the second dataframe
        
        Returns:
            Pandas.Dataframe, Pandas.DataFrame
    """
    return dataframe[:-test_size], dataframe[-test_size:]



if __name__ == "__main__":
    main()