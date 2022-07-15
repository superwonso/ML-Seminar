import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

data = datasets.load_boston()
X, y = shuffle(data.data, data.target, random_state=7)
# Seperate data into training(80%) and testing(20%) sets
num_training = int(0.8*len(X))
x_train, y_train = X[:num_training], y[:num_training]
x_test, y_test = X[num_training:], y[num_training:]
# Create a SVR object(linear kernel)
svr_rbf = SVR(kernel='linear', C=1e0, epsilon=0.1)
# Train the model using the training sets
svr_rbf.fit(x_train, y_train)
# Measure performance of the model
y_test_pred = svr_rbf.predict(x_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print('MSE = ', round(mse,2))
print('EVS = ', round(evs,2))
# Test regression into test data model
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print('\n Predicted price: ', svr_rbf.predict([test_data])[0])
print('\n Actual price: ', test_data[-1])