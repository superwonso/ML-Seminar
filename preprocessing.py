import numpy as np
from sklearn import preprocessing

sample_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3 ,-9.9, -4.5]])

# Binarization
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(sample_data)
print("\nOriginal data:\n", sample_data)
print("\nBinarized data:\n", data_binarized)

# Mean removal
data_mean_removed = preprocessing.scale(sample_data)
print("\nOriginal Mean data:\n", sample_data.mean(axis=0))
print("\nOriginal Standard deviation data:\n", sample_data.std(axis=0))
print("\nMean removed data:\n", data_mean_removed.mean(axis=0))
print("\nMean removed data's Standard deviation data:\n", data_mean_removed.std(axis=0))

# Min-Max scaling
data_minmax_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(sample_data)
print("\nOriginal data:\n", sample_data)
print("\nMin-Max scaled data:\n", data_minmax_scaled)

# L1 normalization
data_l1_normalized = preprocessing.normalize(sample_data, norm='l1')
# L2 normalization
data_l2_normalized = preprocessing.normalize(sample_data, norm='l2')
print("\nL1 normalized data:\n", data_l1_normalized)
print("\nL2 normalized data:\n", data_l2_normalized)