# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import itertools

# %%
df=pd.read_csv("../data/clean_monthly_average_data/monthly_ams_aqi.csv", index_col=['date'],parse_dates=['date'])

# %%
df.shape
df.plot()

# %%
df.describe()

# %%
df.head()

# %%
df.plot()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
for column in df.columns:
    result = seasonal_decompose(df[column])
    result.plot().suptitle(column)
    plt.show()

# %%
df.dtypes

# %%
df2=df.copy()

# %%
df1=df.copy()

# %%
pollutants = df.columns
dataframes = dict()
for pollutant in pollutants:
    dfp = df[[pollutant]].copy()
    dfp[f'Last_year_{pollutant}']=df1[pollutant].shift(12)
    dfp[f'Increase_{pollutant}'] = df1[pollutant] - df1[f'Last_year_{pollutant}']
    dataframes[pollutant] = dfp
    

print(dataframes)
df1.tail()

# %%
y = df2['pm25']
train_size = int(len(y) * 0.6)
train, test = y[0:train_size], y[train_size:len(y)]

# %%
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is not stationary")
        return False
    return True

# %%
if not check_stationarity(train):
    train_diff = train.diff().dropna()
    if not check_stationarity(train_diff):
        print("Differencing did not make the series stationary. Consider using higher order differencing or transformations.")

# %%
best_aic = np.inf 
best_order = None
best_model = None

# %%
for p in range(3):  # example range, can be adjusted
    for d in range(2):  
        for q in range(3):
            try:
                temp_model = ARIMA(train, order=(p,d,q))
                temp_model_fit = temp_model.fit()
                temp_aic = temp_model_fit.aic
                if temp_aic < best_aic:
                    best_aic = temp_aic
                    best_order = (p,d,q)
                    best_model = temp_model_fit
            except:
                continue

print(f"Best ARIMA Order: {best_order}")

# %%
model = pm.auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

print(model.summary())


# %%

model_arima = SARIMAX(train, order=(2,0,0))
model_fit = model_arima.fit()
print(model_fit.summary())

# %%
#SARIMAX MODEL

forecast = model_fit.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast_mean, label='Forecast')
plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.3)
plt.title('SARIMAX Forecast vs Actuals')
plt.legend()
plt.show()
Ensure the train, test, and other variables are appropriately defined in your workspace, and adjust the above code accordingly.







# %%
forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(12,6))
plt.plot(train, label="Training")
plt.plot(test.index, forecast, label="Forecast", color='red')

plt.plot(test, label="Actual", color='green')
plt.legend()
plt.show()

# %%
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# %%
from statsmodels.tsa.stattools import adfuller
result = adfuller(train)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# %%


# %% [markdown]
# TRYING NEW CODE

# %%
data=df.copy()


# %%
data=data[['o3']].copy()

# %%
data.tail()

# %%
data.plot()

# %%
#Seasonal shift 
data['o3_seasonal']=data['o3']-data['o3'].shift(12)
data.head(14)

# %%
#Applying transformations

#data['o3_shift']=data['o3']-data['o3'].shift(1)
data=data.dropna()
data.head()

# %%
data = data.asfreq('M')


# %%
p=d=q=range(0,2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq=[(x[0],x[1],x[2],12)for x in pdq]



# %%
lowest_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(data,
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            
            results = mod.fit()
            
            if results.aic < lowest_aic:
                lowest_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        
        except:
            continue

print('Best SARIMA{}x{}12 - AIC:{}'.format(best_pdq, best_seasonal_pdq, lowest_aic))


# %%
#0,0,1 and 1,1,1,12 are the best parameters for the model

# %%
#Check if data is stationary
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['o3_seasonal'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

#Pvalue is more than 0.05 DATA IS NON STATIONARY 

# %%
data['o3_seasonal'].plot()

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data['o3_seasonal'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(data['o3_seasonal'],lags=40,ax=ax2)

# %% [markdown]
# 

# %%
best_model=SARIMAX(data['o3_seasonal'],order=(0,0,1),seasonal_order=(1,1,1,12))

results=best_model.fit()
print(results.summary().tables[1])

# %%
data['forecast']=results.predict(start='2022-09-30',end='2023-09-30',dynamic=True)
data[['o3_seasonal','forecast']].plot()

# %%
forecast_12_months = results.forecast(steps=12)

forecast_index = [data.index[-1] + pd.DateOffset(months=i) for i in range(1, 13)]
forecast_df = pd.DataFrame(forecast_12_months, index=forecast_index, columns=['forecast'])
combined_data = pd.concat([data, forecast_df], axis=0)
plt.figure(figsize=(12, 6))
combined_data['o3_seasonal'].plot(label='Actual')
combined_data['forecast'].plot(color='red', linestyle='dashed', label='Forecast')
plt.legend()
plt.show()

# %%
forecast_length = 24
forecast_values = results.forecast(steps=forecast_length)

# Create a new DataFrame for the forecasted values
forecast_index = [data.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_length+1)]
forecast_df = pd.DataFrame(forecast_values, index=forecast_index, columns=['forecast'])

# Combine the original data and the forecasted data
combined_data = pd.concat([data, forecast_df], axis=0)

# Plot the actual and forecasted values
plt.figure(figsize=(14, 7))
combined_data['o3_seasonal'].plot(label='Actual')
combined_data['forecast'].plot(color='red', linestyle='dashed', label='Forecast')
plt.legend()
plt.title('Actual vs Forecasted o3 Levels')
plt.xlabel('Date')
plt.ylabel('o3 Level')
plt.grid(True)
plt.show()

# %%
forecast_length = 54

# Generate the forecasted values
forecast_values = results.forecast(steps=forecast_length)

# Create a new DataFrame for the forecasted values
forecast_index = [data.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_length+1)]
forecast_df = pd.DataFrame(forecast_values, index=forecast_index, columns=['forecast'])

# Combine the original data and the forecasted data
combined_data = pd.concat([data, forecast_df], axis=0)

# Plot the actual and forecasted values
plt.figure(figsize=(14, 7))
combined_data['o3'].plot(label='Actual')
combined_data['forecast'].plot(color='red', linestyle='dashed', label='Forecast')
plt.legend()
plt.title('Actual vs Forecasted o3 Levels')
plt.xlabel('Date')
plt.ylabel('o3 Level')
plt.grid(True)

# Ensure the x-axis covers the full date range
plt.xlim(combined_data.index.min(), combined_data.index.max())

plt.show()

# %%
combined_data.head()

# %%



# %%
print(data.index[-1])

# %%
from pandas.tseries.offsets import DateOffset
future_dates=[data.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=data.columns)
future_df = pd.concat([data, future_dates_df])
forecasted_values = results.forecast(steps=24)
future_df.loc[forecasted_values.index, 'forecast'] = forecasted_values.values
future_df[['o3', 'forecast']].plot(figsize=(12, 8))

# %%
results.plot_diagnostics(figsize=(15,12))
plt.show()

# %%
pred=results.get_prediction(start=pd.to_datetime('2023-10-30'),end=pd.to_datetime('2025-12-31'),dynamic=False)
pred_ci=pred.conf_int()
pred_ci

# %%
y_forecasted=pred.predicted_mean
y_truth=data['2023-09-30':]
y_truth['Pred_val']=y_forecasted
y_truth

# %%



