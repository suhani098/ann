import pandas as pd#to give us structured data
import numpy as np#to manupilate complex arrays
from sklearn.impute import SimpleImputer #handles missing values in data
from sklearn.preprocessing import MinMaxScaler#used for normalization 
from sklearn.model_selection import train_test_split#some data is split to training and some for training
from tensorflow import keras

# Read the CSV file 
csv_file_path="C:\\Users\\Suhani\\Desktop\\vscode\\ann\\BTM_hourly.csv"#my file path name
df = pd.read_csv(csv_file_path)  # Replace variable  with the actual path to your CSV file
df=df.apply(pd.to_numeric,errors='coerce')
print(df.head())#printing few samples to check if working properly
print(df.isnull().sum())
df = df.dropna()  # Remove rows with NaN values


#seprating my indipendent and dependent variables in structured manner
x = df.drop('PM2.5', axis=1)
y= df[['PM2.5']]

#preprocessing data 
imputer_x=SimpleImputer(strategy='mean') 
imputer_y=SimpleImputer(strategy='mean')

# calculates the mean of the given coloumn and predicts the misssing
# values using the simpleimputer class that belongs to sklearn.impute
# module.the values are stored in imputer_x instance.
x_imputed=imputer_x.fit_transform(x)
# fits imputer to data and transforms the data with the imputed values
y_imputed=imputer_y.fit_transform(y.values.reshape(-1, 1))
#this is to create datastructure into array and even get coloumn names
x_imputed = pd.DataFrame(x_imputed, columns=x.columns)
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



# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

### normalization of data ###
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
## Normalization of the target variable ###
target_scaler = MinMaxScaler()
y_train_normalized = target_scaler.fit_transform(y_train)
y_test_normalized = target_scaler.transform(y_test)

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train_normalized.shape)
print(X_test.shape)
print(y_test_normalized.shape)

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=7, input_dim=17, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model on the training data
model.fit(X_train, y_train_normalized, batch_size=5, epochs=10, verbose=1)
# Evaluate the model on the testing data
# Evaluate the model on the testing data
evaluation = model.evaluate(X_test, y_test_normalized)
predictions_normalized = model.predict(X_test)
print("Mean Squared Error on Test Data:", evaluation)

# Make predictions on new data
predictions_normalized = model.predict(X_test)
# Inverse transform predictions to the original scale
predictions_original_scale = target_scaler.inverse_transform(predictions_normalized)

# Create a DataFrame to compare predictions with actual values
comparison_df = pd.DataFrame({'Actual': y_test_normalized.flatten(), 'Predicted': predictions_original_scale.flatten()})
print(comparison_df.head())
print(X_train.shape)
print(y_train_normalized.shape)




