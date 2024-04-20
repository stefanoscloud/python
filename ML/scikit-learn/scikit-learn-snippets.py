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

#Evaluate and tune ML models
# Kf= sklearn.model_selection.Kfold(n_splits= 5, shuffle = True) – Creates a cross-validation object that will split the training data into five folds. The shuffle=True argument indicates that the data will be shuffled randomly before being split. 
#Kf.split(X) – Uses the kf object to perform cross-validation on the X dataset, returning the indices with which to split the data into train and test subsets. 
#Skf= sklearn.model_selection.StratifiedKFold(n_splits= 5, shuffle= True) – Creates an object that will return stratified folds based on the percentages of each class provided as labels. The data examples from each class will be shuffled before splitting. 
#Skf.split(X, y) – Performs cross-validation on the X data, while using the data in y as the class labels for stratification. 
#Lpocv= sklearn.model_selection.LeavePOut(p=3) – Creates a cross-validation object using LPOCV, w ith number of examples in test=3
#Loocv= sklearn.model_selection.LeaveOneOut() – Same as sklearn.model_selection.LeavePOut(p=1) 

# Cv= sklearn.model_selection.cross_validate(estimator, X, y, cv=5) - This performs splitting, fitting and evaluation in one step, by running cross validation

