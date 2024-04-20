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


# Linear regression models - LinearRegression() class 
# Model= sklearn.linear_model.LinearRegression() – This constructs a model object with the linear regression algorithm
# Model.fit(X_train, y_train) – Fit a set of training data to the model. The first argument is a data frame or array with the training set (not including labels), whereas the second argument is a data frame or array of just the labels)
# Model.score(x_test, y_test) – Return the coefficient of determination (R^2) score for the model given validation/test data. The first argument is a data frame or array with the validation/test set (not including labels), and the second argument is a data frame or array of just the labels. 
# Model.predict(X_test) – The model makes predictions on the validation/test set, without being given labels.
# Sklearn.metrics.mean_squared_error(y_test, prediction) – Return the MSE of the model, given validation/test data. The second argument is an object created by model.predict(), as discussed above.
# model.coef_ – An attribute that lists the optimal model parameters generated during training. 

# Regularized linear regression models - Ridge(), Lasso(), and ElasticNet() classes
# Model= sklearn.linear_model.Ridge(alpha= 0.1) – This constructs a model object that uses ridge regression (l2 norm). The alpha parameter defines the strength of regularization to apply, which in this case is 0.1
#Model= sklearn.linear_model.lasso(alpha= 0.1) - This constructs a model object that uses lasso regression (l1 norm). Same alpha parameter.
#Model= sklearn.linear_model.ElasticNet(alpha= 0.1, l1_ration= 0.5) – This constructs a model object that uses elastic net regression (a weighted average of both l1 and l2 norms). In addition to using alpha, the object uses the l1_ratio parameter to add more weight to either type of regularization. A value of 0 is the same as ridge and 1 is the same as lasso. 
# You can use these class objects to call the same fit(), score() and predict() methods as with LinearRegression() class. You can also generate the MSE using mean_squared_error() and return the model parameters using the coef_ attribute. 

# Iterative linear regression models - sklearn-linear_model-Ridge() and sklearn.linear_model.SGDRegressor() classes
# Model= sklearn-linear_model-Ridge(alpha= 0.1, solver= ‘sag’) – This constructs a ridge regression object as before, but the solver parameter specified that the algorithm should use SAG, an iterative approach, to minimize cost.
# Model= sklearn.linear_model.SGDRegressor(penalty= ‘l2’, alpha= 0.1, learning_rate= ‘constant’, eta0 = 0.05) 
# Use these classes to call the same methods and attributes as in LinearRegression() class 



