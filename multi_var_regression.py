import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Load data from input file
input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
x,y = data[:,:-1], data[:,-1]
# Devide data into training(80%) and testing(20%) sets
num_training = int(0.8*len(x))
num_test = len(x) - num_training
# Training data
x_train, y_train = x[:num_training], y[:num_training]
# Testing data
x_test, y_test = x[num_training:], y[num_training:]
# Create a linear regression object
linear_regr = linear_model.LinearRegression()
# Train the model using the training sets
linear_regr.fit(x_train, y_train)
# Make predictions using the training set
y_test_pred = linear_regr.predict(x_test)
# Measure performance of the model
print('Linear Regression Performance :')
print('Mean Absolute Error = ', round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print('Mean Squared Error = ', round(sm.mean_squared_error(y_test, y_test_pred), 2))
print('Median absolute error = ', round(sm.median_absolute_error(y_test, y_test_pred), 2))
print('Explained variance score = ', round(sm.explained_variance_score(y_test, y_test_pred), 2))
print('R2 Score = ', round(sm.r2_score(y_test, y_test_pred), 2))
# Create a polynomial regression object
poly_regr = PolynomialFeatures(degree=10)
# Transform the data using polynomial features
x_poly = poly_regr.fit_transform(x_train)
datapoint = [[7.75, 6.35, 5.66]]
poly_datapoint = poly_regr.fit_transform(datapoint)
# Create a linear regression object
poly_linear_regr = linear_model.LinearRegression()
# Train the model using the training sets
poly_linear_regr.fit(x_poly, y_train)
print('\nLinear Regression: \n', linear_regr.predict(datapoint))
print('\nPolynomial Regression: \n', poly_linear_regr.predict(poly_datapoint))