import pandas as pd
import numpy as np

#Prediction
df=pd.read_csv("data/clean_monthly_average_data/monthly_ams_aqi.csv", index_col=['date'],parse_dates=['date'])
print(df.head())
print(df.describe())
print(df.columns)