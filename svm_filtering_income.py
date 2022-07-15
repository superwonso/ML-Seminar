import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Include data file
input_file = 'census-income.data'

# Loading data from a file
x = []
y = []
count_class_1 = 0
count_class_2 = 0
max_data_points = 25000

with open(input_file, 'r') as f:

    for line in f.readlines():
        if count_class_1 >= max_data_points and count_class_2 >= max_data_points:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')
        
        if data[-1] == '<=50K' and count_class_1 < max_data_points:
            x.append(data)
            count_class_1 += 1

        if data[-1] == '>50K' and count_class_2 < max_data_points:
            x.append(data)
            count_class_2 += 1

x = np.array(x)
label_encoder = []
x_encoded = np.empty(x.shape)
for i,item in enumerate(x[0]):
    if item.isdigit():
        x_encoded[:,i] = x[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        x_encoded[:,i] = label_encoder[-1].fit_transform(x[:,i])
x = x_encoded[:,:-1].astype(int)
y = x_encoded[:,-1].astype(int)
# Create SVM classifier
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# Training classifier
classifier.fit(x, y)
# Cross validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(x_train, y_train)
y_test_pred = classifier.predict(x_test)
# Calculate F1 score of the classifier
f1 = cross_val_score(classifier, x, y, scoring='f1_weighted', cv=5)
print('F1 score: {:.2f}%'.format(f1.mean()*100))
# Predict results of trained data points
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0','0', '40', 'United-States']
# Encode test data points
input_data_encoded = [-1] * len(input_data)
input_data_encoded = np.array(input_data_encoded)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = input_data[i]
    else:
        input_data_encoded[i] = label_encoder[count].transform([input_data[i]])
        count += 1
# Run classifier on encoded data points and print result
predicted_class = classifier.predict([input_data_encoded])
print(label_encoder[-1].inverse_transform(predicted_class)[0])
