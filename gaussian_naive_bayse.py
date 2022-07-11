import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from classifier_function import visualize_classifier

input_file = 'data_multivar.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Create Naive Bayes classifier
classifier = GaussianNB()
# Training classifier
classifier.fit(X, y)
# Predict trained data
y_pred = classifier.predict(X)
# Calculate accuracy of classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print('Accuracy of Naive Bayes classifier =', round(accuracy, 2), '%')
# Visualize classifier performance
visualize_classifier(classifier, X, y)
# Split data into training(80%) and test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
# Calculate accuracy of new classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print('Accuracy of New Naive Bayes classifier =', round(accuracy, 2), '%')
# Visualize classifier performance
visualize_classifier(classifier_new, X_test, y_test)
# Calculate cross-validation scores
num_folds = 3
accuracy_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv=num_folds)
print('Accuracy of Naive Bayes classifier using cross validation =', round((100*accuracy_score.mean()), 2), '%')
precision_values = cross_val_score(classifier, X, y, scoring = 'precision_weighted', cv=num_folds)
print('Precision of Naive Bayes classifier using cross validation =', round((100*precision_values.mean()), 2))
recall_values = cross_val_score(classifier, X, y, scoring = 'recall_weighted', cv=num_folds)
print('Recall of Naive Bayes classifier using cross validation =', round((100*recall_values.mean()), 2))
f1_values = cross_val_score(classifier, X, y, scoring = 'f1_weighted', cv=num_folds)
print('F1-score of Naive Bayes classifier using cross validation =', round((100*f1_values.mean()), 2))