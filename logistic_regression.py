import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from classifier_function import visualize_classifier

x = np.array([[3.1,7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],[5.6,5],[3.3,0.4],[3.9,0.9],[2.8,1],[0.5,3.4],[1,4],[0.6,4.9]])
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
# Make logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear',C=0.0000001)
classifier.fit(x, y)
visualize_classifier(classifier, x, y)
