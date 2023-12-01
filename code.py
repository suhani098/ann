#target variable- PM2.5
#predictors - PM10 , NO ,NO2 ,wind speed , wind direction ,SO2 ...etc

#genral data analysis
import pandas as pd #it provides easy data structures and functions needed to manipulate structured data
import numpy as np #for complex numerical computing in data
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
  
# Load data
data = pd.read_csv('BTM_hourly suhani.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']




