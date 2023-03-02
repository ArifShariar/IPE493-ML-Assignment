"""
Author: Arif Shariar Rahman
ID: 1705095
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# importing AAPL data from yfinance
import yfinance as yf

# check if the data is already downloaded
try:
    data = pd.read_csv('AAPL.csv')
except:
    data = yf.download("AAPL", start="2005-01-01", end="2022-01-01")
    data.to_csv('AAPL.csv')

# importing the data
data = pd.read_csv('AAPL.csv')
print(data.head())


# preparing the data
data['next_day_close'] = data['Close'].shift(-1)

# dropping the last row
data = data[:-1]

print("After preparing the data")
print(data.head())

# split dataset in training and testing

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale the features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train five different regression models on the training set and make predictions on the test set
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=10, random_state=0),
    'Support Vector Regressor': SVR(kernel='rbf'),
    'Multi Layer Perceptron Regressor': MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000)
}
models_dict = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'{name}:\nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nMean Absolute Error: {mae}\n')
    # save the result in a dictionary for later use
    models_dict[name] = [mse, rmse, mae]

# save the results in a dataframe
results = pd.DataFrame(models_dict, index=['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'])
print(results)

# save the results in a csv file
results.to_csv('results.csv')

# find the best model based on the lowest RMSE, MAE and MSE
best_model = min(models_dict, key=models_dict.get)
print(f'The best model is {best_model}')


print("Enter the last five business days' closing price:::")
cp_5, cp_4, cp_3, cp_2, cp_1 = map(float, input().split())
input_features = sc.transform([[cp_5, cp_4, cp_3, cp_2, cp_1, 0]])
# make a prediction using the best model
if best_model == 'Linear Regression':
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(input_features)
    print(f'The predicted closing price for the next day is {prediction[0]}')
elif best_model == 'Decision Tree Regressor':
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(input_features)
    print(f'The predicted closing price for the next day is {prediction[0]}')
elif best_model == 'Random Forest Regressor':
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(input_features)
    print(f'The predicted closing price for the next day is {prediction[0]}')
elif best_model == 'Support Vector Regressor':
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    prediction = model.predict(input_features)
    print(f'The predicted closing price for the next day is {prediction[0]}')
elif best_model == 'Multi Layer Perceptron Regressor':
    model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000)
    model.fit(X_train, y_train)
    prediction = model.predict(input_features)
    print(f'The predicted closing price for the next day is {prediction[0]}')

