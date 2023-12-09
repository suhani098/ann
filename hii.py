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
df.replace('None', pd.NA, inplace=True)
print(df.head())



# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


#data preprocessing

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



#normalization
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
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.25, random_state=42)

# Function to generate Deep ANN model 
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(units=3, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time

from tensorflow.keras.callbacks import Callback

class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = {'loss': [], 'val_loss': []}  # Add more metrics if needed

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        # Add more metrics if needed


# Assuming you have defined make_regression_ann somewhere in your code

# Listing all the parameters to try
Parameter_Trials = {'batch_size': [64,128,256],
                    'epochs': [10,20,30,40,50],
                    'Optimizer_trial': ['adam', 'rmsprop']}

# Creating the regression ANN model
RegModel = KerasRegressor(make_regression_ann, verbose=0)

# Defining a custom scoring function for regression
def custom_scoring(orig, pred):
    mae = np.mean(np.abs(orig - pred))
    print('#' * 70, 'Mean Absolute Error:', mae)
    return -mae  # Negative because GridSearchCV looks for the maximum value, and we want to minimize MAE

custom_scorer = make_scorer(custom_scoring, greater_is_better=False)

# Creating the Grid search space
grid_search = GridSearchCV(estimator=RegModel,
                           param_grid=Parameter_Trials,
                           scoring=custom_scorer,
                           cv=5)

# Measuring how much time it took to find the best params
start_time = time.time()

# Running Grid Search for different parameters
grid_search.fit(X_normalized, y_normalized, verbose=1)

end_time = time.time()
print("Total Time Taken: ", round((end_time - start_time) / 60), 'Minutes')

print('Printing Best parameters ')
print(grid_search.best_params_)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = KerasRegressor(make_regression_ann, **best_params, verbose=1)
best_model.fit(X_train, y_train)
test_predictions = best_model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
