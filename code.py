import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
csv_file_path = "C:\\Users\\Suhani\\Desktop\\vscode\\ann\\btm_hourly_2.csv"
df = pd.read_csv(csv_file_path)
df = df.apply(pd.to_numeric, errors='coerce')
# Replace 'None' values with actual NaN values for easier handling
df.replace('None', pd.NA, inplace=True)
print(df.head())


# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Replace 'None' values with actual NaN values for easier handling
df.replace('None', pd.NA, inplace=True)

# Separating features and target variable
X = df.drop('PM2.5', axis=1)  # Features
y = df['PM2.5']  # Target variable

# Creating SimpleImputer instance to impute missing values with the mean
imputer_x = SimpleImputer(strategy='most_frequent')

# Creating SimpleImputer instance to impute missing values with the mean for y
imputer_y = SimpleImputer(strategy='most_frequent')

# Impute missing values in the features
X_imputed = imputer_x.fit_transform(X)

# Reshape y to a 2D array
y = y.values.reshape(-1, 1)
y_imputed = imputer_y.fit_transform(y)
print(X_imputed)
print(y_imputed)

from sklearn.preprocessing import MinMaxScaler

# Assuming you have your data in X and y
# X is the input features, and y is the target variable

# Create MinMaxScaler objects
PredictorScaler = MinMaxScaler()
TargetVarScaler = MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X_imputed)
TargetVarScalerFit = TargetVarScaler.fit(y_imputed.reshape(-1, 1))  # Reshape y if it's a 1D array

# Generating the standardized values of X and y
X_normalized = PredictorScalerFit.transform(X_imputed)
y_normalized = TargetVarScalerFit.transform(y_imputed.reshape(-1, 1)).flatten()  # Reshape back to 1D array
print(X_normalized)
print(y_normalized)
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Function to generate Deep ANN model 
# Function to generate Deep ANN model 
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(units=5, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model

###########################################
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Listing all the parameters to try
Parameter_Trials={'batch_size':[30,60,90],
                      'epochs':[10,20],
                    'Optimizer_trial':['adam', 'rmsprop']
                 }

# Creating the regression ANN model
RegModel=KerasRegressor(make_regression_ann, verbose=0)

###########################################
from sklearn.metrics import make_scorer

# Defining a custom function to calculate accuracy
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

#########################################
# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search=GridSearchCV(estimator=RegModel, 
                         param_grid=Parameter_Trials, 
                         scoring=custom_Scoring, 
                         cv=5)

#########################################
# Measuring how much time it took to find the best params
import time
StartTime=time.time()

# Running Grid Search for different paramenters
grid_search.fit(X_normalized,y_normalized, verbose=1)

EndTime=time.time()
print("########## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes')

print('### Printing Best parameters ###')
grid_search.best_params_