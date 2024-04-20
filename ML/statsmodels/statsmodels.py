# Statsmodels package
# Univariate forecasting models
# statsmodels package includes a tsa.arima.model module which includes the ARIMA() class 


# Model= statsmodel.tsa.arima.model.ARIMA(y_train, order= (3, 1, 1), seasonal_order= (3, 1, 1, 12)) – This constructs an ARIMA model object with a  p of 3 and a   d   and q of 1. These same values are incorporated into the model’s seasonality, which also uses 12 as the periodicity. 
# Model.predict(start= ‘2023-01-01 00:00:00’, end= ‘2023-06-01 00:00:00’) – Use the model to make predictions for the next time intervals that fall between start and end. In this example, the interval is one month, so the model will make six total predictions.
