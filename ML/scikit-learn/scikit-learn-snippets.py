# Scikit-learn Python library

# Feature engineering
# scaler= sklearn.preprocessing.MinMaxScaler() – Create an object that will be used for normalization (min-max scaling)
# scaler= sklearn.preprocessing.StandardScaler() – Create an object that will be used for standardization
# scaler.fit_transform(data) – Return the scaled values of all input data based on an existing scaler object.
# binner = sklearn.preprocessing.KBinsDisretizer(n_bins=4, encode = ‘ordinal’, strategy ‘ ‘uniform’) – Create an object that will place values into the number of specified bins using an ordinal integer to identify each bin (starting at 0 and incrementing by 1). The ‘uniform” strategy is equal-width binning. Use ‘quantile’ for equal-frequency binning.
# binner.fit_transform(data) – Return the binned values of all input data based on an existing binner object.
# pca= sklearn.decomposition.PCA(n_components = 2) – Create an object that will perform principal component analysis (PCA) on a dataset and return two features.
# Pca.fit_transform(data) – Return the reduced feature values from the input data, based on an existing pca object.

#Train ML models
# sklearn.model_selection module with the sklearn.model_selection.train_test_split() 
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 1)
