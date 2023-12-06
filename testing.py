# Reading the cleaned numeric car prices data
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense



# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

# Read the CSV file
csv_file_path = "C:\\Users\\Suhani\\Desktop\\vscode\\ann\\BTM_hourly.csv"
df = pd.read_csv(csv_file_path)
df = df.apply(pd.to_numeric, errors='coerce')#used to convert the strings present in dtataframe to numeric
print(df.head())


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

print(df.isnull().sum())
df = df.dropna()


# Separate Target Variable and Predictor Variables
Predictors=imputer_x.drop('PM2.5', axis=1)
TargetVariable= imputer_x[['PM2.5']]

# Check the content and column names of Predictors
print(Predictors.head())  # Print the first few rows
print(Predictors.columns)  # Print column names

### normalization of data of data ###
PredictorSnorm = MinMaxScaler()
TargetVarnorm=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorSnorm.fit(Predictors)
TargetVarScalerFit=TargetVarnorm.fit(TargetVariable.values.reshape(-1, 1))
 
# Generating the normalized values of X and y
X=PredictorScalerFit.transform(Predictors)
y=TargetVarScalerFit.transform(TargetVariable)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to generate Deep ANN model 
def make_regression_ann(Optimizer_trial):
    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
    return model


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Listing all the parameters to try
Parameter_Trials={'batch_size':[10,20,30,40,50],
                      'epochs':[10,20,30,40,50],
                    'Optimizer_trial':['adam', 'rmsprop']
                 }

# Creating the regression ANN model
RegModel=KerasRegressor(make_regression_ann, verbose=0)

from sklearn.metrics import make_scorer

# Defining a custom function to calculate accuracy
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search=GridSearchCV(estimator=RegModel, 
                         param_grid=Parameter_Trials, 
                         scoring=custom_Scoring, 
                         cv=5)

# Measuring how much time it took to find the best params
import time
StartTime=time.time()

# Running Grid Search for different paramenters
grid_search.fit(X,y, verbose=1)

EndTime=time.time()
print("########## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes')

print('### Printing Best parameters ###')
grid_search.best_params_