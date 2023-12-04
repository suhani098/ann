import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
csv_file_path = "C:\\Users\\Suhani\\Desktop\\vscode\\ann\\BTM_hourly.csv"
df = pd.read_csv(csv_file_path)
df = df.apply(pd.to_numeric, errors='coerce')
print(df.head())
print(df.isnull().sum())
df = df.dropna()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Preprocessing data

# Calculate the mean of each column and impute missing values
imputer_x = SimpleImputer(strategy='mean')
imputer_y = SimpleImputer(strategy='mean')
imputer_x = pd.DataFrame(imputer_x.fit_transform(df), columns=df.columns)
imputer_y = pd.DataFrame(imputer_y.fit_transform(df[['PM2.5']].values.reshape(-1, 1)), columns=['PM2.5'])

# Handle outliers (you can use a more sophisticated method based on your data)
# Assuming 'PM2.5' is the target variable
lower_threshold = imputer_x['PM2.5'].quantile(0.05)
upper_threshold = imputer_x['PM2.5'].quantile(0.95)

imputer_x['PM2.5'] = np.where(imputer_x['PM2.5'] > upper_threshold, upper_threshold, imputer_x['PM2.5'])
imputer_x['PM2.5'] = np.where(imputer_x['PM2.5'] < lower_threshold, lower_threshold, imputer_x['PM2.5'])

# Split the data into features (X) and target variable (y)
X = imputer_x.drop('PM2.5', axis=1)
y = imputer_x['PM2.5']

# Normalize data
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
y_normalized = min_max_scaler.fit_transform(y.values.reshape(-1, 1))

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=16, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=9, kernel_initializer='normal', activation='relu'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(4, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model on the training data
model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared (R2) Value: {r2:.4f}')

# Calculate Accuracy
accuracy = (np.abs(y_pred - y_test) < 0.1).mean()  # Assuming a tolerance of 0.1, adjust as needed
print(f'Accuracy: {accuracy:.4f}')

# Create a DataFrame with actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

# Display the DataFrame in the terminal
print(comparison_df)
