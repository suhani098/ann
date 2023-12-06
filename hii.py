#import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox

# Read the CSV file
csv_file_path = "C:\\Users\\Suhani\\Desktop\\vscode\\ann\\btm_hourly_2.csv"
df = pd.read_csv(csv_file_path)
df = df.apply(pd.to_numeric, errors='coerce')
df[df < 0] = None#replces all the negetive value by none 
df.replace('None', pd.NA, inplace=True)
print(df.head())


#to check if we have negetive values and number of negetive values
has_negative_values = (df < 0).any().any()
if has_negative_values:
    print("The dataset contains negative values.")
else:
    print("The dataset does not contain negative values.")
num_negative_values = (df < 0).sum().sum()
print(f"The dataset contains {num_negative_values} negative values.")


# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


#data preprocessing

# Separating features and target variable
X = df.drop('PM2.5', axis=1)  # Features
y = df['PM2.5']  # Target variable

#preprocessing data 
#imputation
imputer_x=SimpleImputer(strategy='mean') 
imputer_y=SimpleImputer(strategy='mean')
# calculates the mean of the given coloumn and predicts the misssing
x_imputed=imputer_x.fit_transform(X)
# fits imputer to data and transforms the data with the imputed values
y_imputed=imputer_y.fit_transform(y.values.reshape(-1, 1))

#to check imputed values
#this is to create datastructure into array and even get coloumn names
x_imputed = pd.DataFrame(x_imputed, columns=y.values.columns)
y_imputed = pd.DataFrame(y_imputed, columns=['PM2.5'])
#create an seprate csv file to write your imputed data into it 
x_imputed.to_csv('imputed_x.csv', index=False)
y_imputed.to_csv('imputed_y.csv', index=False)
# Assuming you have two CSV files: 'file1.csv' and 'file2.csv'
file1_path = 'C:\\Users\\Suhani\\Desktop\\vscode\\imputed_x.csv'
file2_path = 'C:\\Users\\Suhani\\Desktop\\vscode\\imputed_y.csv'
# Read the CSV files into DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)


