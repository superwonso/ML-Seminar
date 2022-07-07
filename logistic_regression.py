import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def visualize_classifier(classifier, x, y):
    min_x, max_x = x[:,0].min() - 1.0, x[:,0].max() + 1.0
    min_y, max_y = x[:,1].min() - 1.0, x[:,1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    plt.scatter(x[:,0], x[:,1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))
    plt.show()

x = np.array([[3.1,7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],[5.6,5],[3.3,0.4],[3.9,0.9],[2.8,1],[0.5,3.4],[1,4],[0.6,4.9]])
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
# Make logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=100000)
classifier.fit(x, y)
visualize_classifier(classifier, x, y)
