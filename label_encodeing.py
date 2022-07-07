import numpy as np
from sklearn import preprocessing

sample_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
# Make label encoder and training
encoder = preprocessing.LabelEncoder()
encoder.fit(sample_labels)
# Print label mapping
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
# Encode label set
test_labels = ['green', 'red', 'black']
encoded_labels = encoder.transform(test_labels)
print("\n Labels =", test_labels)
print("\nEncoded values:", encoded_labels)
# Decode random number set
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values:", encoded_values)
print("\nDecoded labels:", list(decoded_list))