import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template_string, render_template, request, jsonify
from sqlalchemy import create_engine, text, inspect, Table
import requests
import my_prediction
from datetime import datetime


def rename_column(df):
    """
    Modify column names in the DataFrame by removing leading spaces.

    Parameters:
    - df (pandas.DataFrame): The DataFrame whose columns need to be renamed.

    Returns:
    None
    """

    new_column_names = {' pm25': 'pm25',
                       ' pm10': 'pm10',
                       ' o3': 'o3',
                       ' no2': 'no2',
                       ' so2': 'so2',
                       ' co': 'co'
    }
    
    df.rename(columns=new_column_names, inplace=True)

def reformat_record(df): 
    """
    Reformat the input DataFrame by correcting the date format, setting the date as the index,
    sorting the dataset by the date index, and changing the data type of rows to float.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be reformatted.

    Returns:
    pandas.DataFrame: The reformatted DataFrame.
    """

    df['date'] = pd.to_datetime(df['date']) 
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True) 
    df = df.apply(pd.to_numeric, errors='coerce') 
    return df

def data_imputation(df):
    """
    Fill missing values in specified columns using forward-fill and mean imputation.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame for imputation.

    Returns:
    None (modifies the input DataFrame in-place).
    """

    columns_to_fill = ['pm25', 'pm10', 'o3', 'no2'] 
    
    for column in columns_to_fill:
        df[column].fillna(method='ffill', inplace=True) 
    
    for column in columns_to_fill:
        df[column].fillna(df[column].mean(), inplace=True) 

def data_transformation(df):
    """
    Calculate monthly averages of pm25, pm10, o3, and no2.
    Return a DataFrame containing monthly averages.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame for transformation.

    Returns:
    pandas.DataFrame: A DataFrame containing monthly averages for specified columns.
    """
    
    columns_to_average = ['pm25', 'pm10', 'o3', 'no2'] # specify the columns needed to be aggregated
    monthly_averages_df = pd.DataFrame() # init a empty dict for further appending 
    for column in columns_to_average:
        monthly_averages = df[column].resample('M').mean().round(2) # calculate the average value in each month
        monthly_averages_df[column] = monthly_averages # append a new column in the dictioary
        monthly_averages_df[column].fillna(monthly_averages_df[column].mean().round(2), inplace=True)
    return monthly_averages_df

def _load_data_to_db():
    """
    Load processed monthly air quality index(AQI) data into PostgreSQL database.

    This function connects to a PostgreSQL database, drops existing tables, and replaces them
    with new table created from CSV files containing monthly AQI data for different cities.

    Returns: 
    None
    """

    # Connect to PostgreSQL database
    engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")

    # Define the name of tables to be created in the database
    table_names = ["AMS_AQI", "BXL_AQI", "HEL_AQI", "LDN_AQI", "PAR_AQI"]

    # Drop existing tables in the database
    with engine.connect() as conn:
        for table_name in table_names:
            query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
            result = conn.execute(text(query))

    # Read monthly AQI data from processed csv files
    monthly_ams_aqi_df = pd.read_csv("../data/preprocessed_monthly_data/amsterdam_imputated.csv", index_col=0, parse_dates=True)
    monthly_bxl_aqi_df = pd.read_csv("../data/preprocessed_monthly_data/brussels_imputated.csv", index_col=0, parse_dates=True)
    monthly_hel_aqi_df = pd.read_csv("../data/preprocessed_monthly_data/helsinki_imputated.csv", index_col=0, parse_dates=True)
    monthly_ldn_aqi_df = pd.read_csv("../data/preprocessed_monthly_data/london_imputated.csv", index_col=0, parse_dates=True)
    monthly_par_aqi_df= pd.read_csv("../data/preprocessed_monthly_data/paris_imputated.csv", index_col=0, parse_dates=True)

    # Write dataframes to SQL tables in the database
    monthly_ams_aqi_df.to_sql(table_names[0], engine, if_exists="replace", index=True)
    monthly_bxl_aqi_df.to_sql(table_names[1], engine, if_exists="replace", index=True)
    monthly_hel_aqi_df.to_sql(table_names[2], engine, if_exists="replace", index=True)
    monthly_ldn_aqi_df.to_sql(table_names[3], engine, if_exists="replace", index=True)
    monthly_par_aqi_df.to_sql(table_names[4], engine, if_exists="replace", index=True)

def _fetch_data_from_db(city_AQI):
    """
    Fetch monthly air quality index (AQI) data for a specific city from a PostgreSQL database.

    Parameters:
    - city_AQI (str): The name of the city's AQI data table in the database.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the fetched monthly AQI data with the 'date' column set as the index.
    """

    # Connect to PostgreSQL database
    engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")
    
    # Read the specified city's AQI data table from the database into a Dataframe
    monthly_data_table = pd.read_sql_table(city_AQI, engine, index_col='date')

    return monthly_data_table

def generate_comparison_bar_chart():
    """
    Generate a bar chart comparing air pollutant levels (PM2.5, PM10, O3, NO2) for multiple cities.

    This function fetches the latest air quality index (AQI) data for five cities (Amsterdam, Brussels, Helsinki, London, Paris)
    from a PostgreSQL database, constructs a DataFrame, and creates an interactive bar chart for pollutant comparison.

    Returns:
    str: HTML code representing the interactive bar chart.

    Dependencies:
    - _fetch_data_from_db: A function to fetch AQI data from the database.
    - pandas: Data manipulation library.
    - plotly.graph_objects: Plotly library for creating interactive plots.
    """
    ams_monthly_aqi = _fetch_data_from_db("AMS_AQI")
    bxl_monthly_aqi = _fetch_data_from_db("BXL_AQI")
    hel_monthly_aqi = _fetch_data_from_db("HEL_AQI")
    ldn_monthly_aqi = _fetch_data_from_db("LDN_AQI")
    par_monthly_aqi = _fetch_data_from_db("PAR_AQI")

    cities = ['Amsterdam', 'Brussels', 'Helsinki', "London", "Paris"]

    city_data = {
        'Amsterdam': ams_monthly_aqi.iloc[-1].to_list(),
        'Brussels': bxl_monthly_aqi.iloc[-1].to_list(),
        'Helsinki': hel_monthly_aqi.iloc[-1].to_list(),
        'London': ldn_monthly_aqi.iloc[-1].to_list(),
        'Paris': par_monthly_aqi.iloc[-1].to_list(),
    }
    df_list = []     

    for city in cities:
        city_data_for_df = {
            'City': city,
            'PM25': city_data[city][0],
            'PM10': city_data[city][1],
            'O3': city_data[city][2],
            'NO2': city_data[city][3],
        }
        df_list.append(pd.DataFrame([city_data_for_df]))

    df_comparison = pd.concat(df_list, ignore_index=True)

    df_comparison.set_index('City', inplace=True)

    fig = go.Figure()

    for column in df_comparison.columns.to_list():
        temp_df = df_comparison.sort_values(by=column, ascending=False)
        fig.add_trace(
            go.Bar(
                x = temp_df.index,
                y = temp_df[column],
                name = column
            )
        )
    
    fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list(
            [dict(label = 'PM2.5',
                method = 'update',
                args = [{'visible': [True, False, False, False]},
                        {'title': 'PM2.5 Pollution Comparison by City (Last Value)',
                        'showlegend':True}]),
            dict(label = 'PM10',
                method = 'update',
                args = [{'visible': [False, True, False, False]}, 
                        {'title': 'PM10 Pollution Comparison by City (Last Value)',
                        'showlegend':True}]),
            dict(label = 'O3',
                method = 'update',
                args = [{'visible': [False, False, True, False]}, 
                        {'title': 'O3 Pollution Comparison by City (Last Value)',
                        'showlegend':True}]),
            dict(label = 'NO2',
                method = 'update',
                args = [{'visible': [False, False, False, True]}, 
                        {'title': 'NO2 Pollution Comparison by City (Last Value)',
                        'showlegend':True}])
                            ])
                        )]
    )

    plot_html = fig.to_html(full_html=False)
    return plot_html

def generate_world_map():
    """
    Generate a world map with markers for selected cities.

    This function creates an interactive world map using Plotly Express, with markers representing
    selected cities (Amsterdam, Brussels, Helsinki, London, Paris). Each marker is colored and sized
    based on predefined attributes (color and size).

    Returns:
    str: HTML code representing the interactive world map.

    Dependencies:
    - plotly.express: Plotly library for creating interactive plots.
    """
    cities = [
        {'name': 'Amsterdam', 'lat': 52.379189, 'lon': 4.899431, 'color': 'red', 'size': 100},
        {'name': 'Brussels', 'lat': 50.850346, 'lon': 4.351721, 'color': 'blue', 'size': 100},
        {'name': 'Helsinki', 'lat': 60.169856, 'lon': 24.938379, 'color': 'green', 'size': 100},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'color': 'purple', 'size': 100},
        {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522, 'color': 'orange', 'size': 100}
    ]

    fig = px.scatter_geo(cities, lat='lat', lon='lon', text='name', color='color', size='size')
    fig.update_geos(
        projection_type="natural earth",
        center=dict(lon=4.899431, lat=52.379189),
        scope="europe")
    fig.update_layout(showlegend=False)
    plot_html = fig.to_html(full_html=False)

    return plot_html

def generate_historical_data(city):
    """
    Generate an interactive plot displaying historical and forecasted air pollutant data for a specific city.

    This function fetches historical air quality index (AQI) data for a given city from a PostgreSQL database.
    It then plots the historical data, the current concentration, and the forecasted values for PM2.5, PM10, O3, and NO2.

    Parameters:
    - city (str): The name of the city for which historical data is to be generated.

    Returns:
    str: HTML code representing the interactive plot.

    Dependencies:
    - _fetch_data_from_db: A function to fetch AQI data from the database.
    - _get_realtime_data: A function to fetch real-time data for the specified city.
    - my_prediction: A module providing forecasting functionality.
    - pandas: Data manipulation library.
    - plotly.graph_objects: Plotly library for creating interactive plots.
    
    """
    if city == "Amsterdam":
        targeted_data = _fetch_data_from_db("AMS_AQI")
    elif city == "Brussels":
        targeted_data = _fetch_data_from_db("BXL_AQI")
    elif city == "Helsinki":
        targeted_data = _fetch_data_from_db("HEL_AQI")
    elif city == "London":
        targeted_data = _fetch_data_from_db("LDN_AQI")
    else:
        targeted_data = _fetch_data_from_db("PAR_AQI")
    
    pm25 = targeted_data["pm25"]
    pm10 = targeted_data["pm10"]
    o3 = targeted_data["o3"]
    no2 = targeted_data["no2"]


    df_pollutants = pd.DataFrame({
        'PM25': pm25,
        'PM10': pm10,
        'O3': o3,
        'NO2': no2,
    })

    trace_buttons = []
    traces = []
    forecast_buttons = []
    targeted_data['ds'] = df_pollutants.index
    current_values = pd.DataFrame(_get_realtime_data(city)).rename(columns={'PM2.5':'PM25'})
    
    print(current_values)
    current_buttons= []
    for column in df_pollutants.columns:
        traces.append(
            go.Scatter(
                x = df_pollutants.index,
                y = df_pollutants[column],
                visible=True,
                name = column
            )
        )
        trace_buttons.append(
            dict(
            method='restyle',
            label=column,
            visible=True,
            args=[{'visible':True},[i for i,trace in enumerate(traces) if trace.name == column]],
            args2=[{'visible':False},[i for i,trace in enumerate(traces) if trace.name == column]]
        )
        )
        traces.append(
            go.Scatter(
                x = [datetime.now()],
                y = [current_values[column]['concentration']],
                visible=False,
                name= "current "+column
            )
        )
        current_buttons.append(
            dict(
                method='restyle',
                label=column,
                visible=True,
                args=[{'visible':True},[i for i, trace in enumerate(traces) if trace.name == "current "+column]],
                args2= [{'visible':False},[i for i, trace in enumerate(traces) if trace.name == "current "+column]]
            )
        )
        targeted_data['y'] = df_pollutants[column]
        forecast = my_prediction.get_forecast(city[:2],column,data=targeted_data)
        traces.append(
            go.Scatter(
                x = forecast['ds'],
                y = forecast['yhat'],
                visible=False,
                name = column+" forecast"
            )
        )
        forecast_buttons.append(
            dict(
            method='restyle',
            label=column,
            visible=True,
            args=[{'visible':True},[i for i, trace in enumerate(traces) if trace.name == column+" forecast"]],
            args2=[{'visible':False},[i for i, trace in enumerate(traces) if trace.name == column+" forecast"]]
        )
        )
            
    forecast_button = [
        dict(
            method='restyle',
            label='Forecasts',
            visible=True,
            args=[{'visible':False},[i for i, trace in enumerate(traces) if trace.name.endswith(" forecast")]],
            args2=[{'visible':True},[i for i, trace in enumerate(traces) if trace.name.endswith(" forecast")]]
        )
    ]
    current_button = [
        dict(
            method='restyle',
            label='Current Concentration',
            visible=True,
            args=[{'visible':False},[i for i, trace in enumerate(traces) if trace.name.startswith("current ")]],
            args2=[{'visible':True},[i for i, trace in enumerate(traces) if trace.name.startswith("current ")]]
        )
    ]
    all_button = [
        dict(
            method='restyle',
            label='Pollutant',
            visible=True,
            args=[{'visible':True},[i for i, trace in enumerate(traces) if not trace.name.endswith(" forecast")]],
            args2=[{'visible':False},[i for i, trace in enumerate(traces) if not trace.name.endswith(" forecast")]]
    )]

    layout = go.Layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                x=0.99,
                y=1.39,
                showactive=True,
                buttons=all_button+trace_buttons,
                xanchor='right',
                yanchor='top'
            ),
            dict(
                type='buttons',
                direction='right',
                x=0.99,
                y=1.130,
                showactive=True,
                buttons=forecast_button+forecast_buttons,
                xanchor='right',
                yanchor='top'
            ),
            dict(
                type='buttons',
                direction='right',
                x=0.99,
                y=1.26,
                showactive=True,
                buttons=current_button+current_buttons,
                xanchor='right',
                yanchor='top'
            )
        ],
        showlegend=True
    )
    fig = go.Figure(data=traces,layout=layout)

    plot_html = fig.to_html(full_html=False)
    return plot_html


def generate_realtime_data_table(city):
    """
    Generate an HTML table displaying real-time air quality data for a specific city.

    This function fetches real-time air quality data for a given city and creates a table using Plotly Express.
    The table includes information on pollutant concentrations and AQI values for NO2, O3, PM2.5, and PM10.

    Parameters:
    - city (str): The name of the city for which real-time data is to be displayed.

    Returns:
    str: HTML code representing the table.
    """
    
    data = _get_realtime_data(city)
    clear_data = ['NO2', 'O3', 'PM2.5', 'PM10']
    filtered_data = {key: data[key] for key in clear_data if key in data}
        

    # Extract keys and values from filtered_data
    parameters = list(filtered_data.keys())
    concentrations = [data['concentration'] for data in filtered_data.values()]
    aqi_values = [data['aqi'] for data in filtered_data.values()]

    # Create a Plotly table
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Pollutant', 'Concentration', 'AQI']),
    cells=dict(values=[parameters, concentrations, aqi_values])
        )])

    # Update layout
    fig.update_layout(title=city)

    table_html = fig.to_html(full_html=False)
    return table_html

def _get_realtime_data(city):
    """
    Fetch real-time air quality data for a specific city from an external API.

    This function makes requests to an external API to retrieve real-time air quality data for a specified city.
    It iterates over a list of API keys in case one key fails to authenticate the request.

    Parameters:
    - city (str): The name of the city for which real-time data is to be fetched.

    Returns:
    dict or bool: A dictionary containing real-time air quality data if successful, False if unsuccessful.

    Dependencies:
    - _fetch_api_data: A helper function to fetch data from the external API.
    """

    url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"

    # Query parameters with the specified city
    querystring = {"city":city}

    api_keys = [
        "Your API_KEY"
        ]
    
    for api_key in api_keys:
        try:
            return _fetch_api_data(querystring,api_key)
        except requests.HTTPError:
            continue

    return False     

def _fetch_api_data(querystring,api_key,url="https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality",api_host="air-quality-by-api-ninjas.p.rapidapi.com"):
    """
    Fetch data from an external API using the specified query parameters and API key.

    This function makes a GET request to an external API using the provided URL, query parameters, API key,
    and API host. It handles HTTP errors and returns the response data in JSON format.

    Parameters:
    - querystring (dict): Query parameters to be included in the API request.
    - api_key (str): API key for authentication.
    - url (str): URL of the API endpoint. Default is the air quality API endpoint.
    - api_host (str): Hostname to be included in the request headers. Default is the air quality API host.

    Returns:
    dict: JSON response data from the API.
    """
    headers={
        "X-RapidAPI-Key" : api_key,
        "X-RapidAPI-Host" : api_host
    }
    response = requests.get(url,headers=headers,params=querystring)

    if not response.status_code == requests.codes.ok:
        print("Error:", response.status_code, response.text)
        response.raise_for_status()
    
    return response.json()

def target_measurement(city, pollutant):
    """
    Generate a gauge chart indicating the current measurement of a pollutant(the latest month) compared to the target threshold.

    This function fetches the latest(the latest month) air quality index (AQI) data for a specified city and pollutant
    from a PostgreSQL database. It then creates a gauge chart using Plotly Express to visually represent
    the current measurement in comparison to the target threshold.

    Parameters:
    - city (str): The name of the city for which AQI data is to be fetched.
    - pollutant (str): The pollutant for which the measurement is to be visualized.

    Returns:
    str: HTML code representing the gauge chart.

    Dependencies:
    - _fetch_data_from_db: A function to fetch AQI data from the database.
    - plotly.graph_objects: Plotly library for creating interactive plots.
    """

    threshold_dict = {
        "Amsterdam": {"pm25": 5,
                      "pm10": 15,
                      "o3": 60,
                      "no2": 10},
        "Brussels": {"pm25": 5,
                      "pm10": 15,
                      "o3": 60,
                      "no2": 10},
        "Helsinki": {"pm25": 5,
                      "pm10": 15,
                      "o3": 60,
                      "no2": 10},
        "London": {"pm25": 5,
                      "pm10": 15,
                      "o3": 60,
                      "no2": 10},
        "Paris": {"pm25": 5,
                      "pm10": 15,
                      "o3": 60,
                      "no2": 10}
    }
    if city == "Amsterdam":
        target = _fetch_data_from_db("AMS_AQI")
    elif city == "Brussels":
        target = _fetch_data_from_db("BXL_AQI")
    elif city == "Helsinki":
        target = _fetch_data_from_db("HEL_AQI")
    elif city == "London":
        target = _fetch_data_from_db("LDN_AQI")
    else:
        target = _fetch_data_from_db("PAR_AQI")

    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=target[pollutant].iloc[-1],  # Replace with your actual value
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": pollutant},
    gauge={
        "axis": {"range": [0, 100]},  # Define the range of values
        "steps": [
            {"range": [0, 25], "color": "red"},
            {"range": [25, 50], "color": "orange"},
            {"range": [50, 75], "color": "lightgreen"},
            {"range": [75, 100], "color": "green"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 4},
            "thickness": 0.75,
            "value": threshold_dict[city][pollutant],  # Replace with your threshold value
        },
    }
))
    # fig.show()
    plot_html = fig.to_html(full_html=False)
    return plot_html

# Load the data into the database
_load_data_to_db()

# Initialize the Flask application
app = Flask(__name__)


@app.route('/')
def index():
    # Generate content for the world map and comparison bar chart
    world_map_html = generate_world_map()
    comparison_bar_chart_html = generate_comparison_bar_chart()

    # Render the 'index.html' template with the generated content
    return render_template('index.html',
                           world_map_content_html=world_map_html,
                           historical_data_comparison_html=comparison_bar_chart_html)
    


@app.route('/city', methods=['POST'])
def replacement_test():
    # Extract data from the JSON payload of the POST request
    data = request.get_json()
    selected_value = data.get('selectedValue')

    # Generate HTML content using various functions
    historical_data_html = generate_historical_data(selected_value) 
    realtime_data_content_html = generate_realtime_data_table(selected_value)
    pm25_gauge_html = target_measurement(selected_value, "pm25")
    pm10_gauge_html = target_measurement(selected_value, "pm10")  
    o3_gauge_html = target_measurement(selected_value, "o3")
    no2_gauge_html = target_measurement(selected_value, "no2") 

    # Create a dictionary containing the generated HTML content 
    response_data = {
        "historical_data": historical_data_html,
        "realtime_data_content": realtime_data_content_html,
        "pm25_gauge": pm25_gauge_html,
        "pm10_gauge": pm10_gauge_html,
        "o3_gauge": o3_gauge_html,
        "no2_gauge": no2_gauge_html
    }  

     # Return the JSON response containing the generated HTML content
    return response_data

if __name__ == '__main__':
    app.run(debug=True)
