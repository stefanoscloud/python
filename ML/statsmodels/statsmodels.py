# Statsmodels package

# Univariate forecasting (regression) models - tsa.arima.model module which includes the ARIMA() class 
# Model= statsmodel.tsa.arima.model.ARIMA(y_train, order= (3, 1, 1), seasonal_order= (3, 1, 1, 12)) – This constructs an ARIMA model object with a  p of 3 and a   d   and q of 1. These same values are incorporated into the model’s seasonality, which also uses 12 as the periodicity. 
# Model.predict(start= ‘2023-01-01 00:00:00’, end= ‘2023-06-01 00:00:00’) – Use the model to make predictions for the next time intervals that fall between start and end. In this example, the interval is one month, so the model will make six total predictions.

# Multivariate forecasting models - tsa.vector_ar.var_model module which includes the VAR() class 
# Model= statsmodels.tsa.vector_ar.var_model.VAR(train_endo, train_exo) – This constructs a VAR model object with a set of endogenous and exogenous variables. 
#Model_fit= model.fit(3) – This fits the VAR model with 3 as the maximum number of lags. 

# You can use this VAR() class object to call the same score() method as you would with other regression models. You can also call summary() to get a detailed output of the model’s attributes, including the coefficients, error scores, and distribution properties.
# Lags= model_fit.k_ar – Use this attribute to retrieve the number of lags in the fit model. 
# Model.predict(model_fit.params, start= lags, end= lags+5, lags=lags) – Use the model parameters to make predictions for the next time intervals that fall between start and end. In this example, start is equal to the number of lags and end is the next five lags ahead of that (six predictions). The lags argument also takes the number of lags. 
# Model_fit.forecast(y= train_endo[-lags:], steps= 6) – An alternative method for forecasting the next number of specified steps. The y argument slices the last number of rows from the training set, where that number is equal to the number of lags. 

# Test stationarity - tsa.stattools module
#Statsmodels.tsa.stattools.adfuller(train_var) – This returns a tuple of various results of the ADF test on the provided variable. The second value in the tuple is the p value. 
#Statsmodels.tsa.stattools(train_var) – This returns a tuple of various results of the KPSS test on the provided variable. The second value in the tuple is the p value. 

