# Pandas DataFrame() class

# Load data
# df= pandas.read_csv(‘path to csv’) – Pass in the path of a CSV to load it into the Python environment as DF
# df= pandas.read_excel(‘path to excel’) – Pass in the path of an Excel to load it into the Python environment as DF
# df= pandas.read_pickle(‘path to pickle’) – Pass in the path of a pickle file to load it into the Python environment as DF
# df.info() – Retries various attributes of the data frame, including the number of entries (rows), number of columns, the name of each column and the data type of each column.
# df.head() – Return the first five rows of a data frame.

# Transform data
# pandas.to_numeric(‘10’) – Convert a  non-numeric data type to a  numeric data type, if possible.
# pandas.to_datetime(‘December 10, 2023, 9:15 AM’) – Convert a string into a proper datetime format. Supports various formatting ways for dates and times.
# pandas.DataFrame.duplicated(df) – Return a serries of Booleans that indicate which row numbers or labels in a df have duplicate values. The first occurrence is false and all others are true.
# pandas.DataFrame.isna(df) - Return a data frame of Booleans that indicates which data values are formatted as a missing data type (na)
# pandas.DataFrame.notna(df) – Return a data frame of Booleans that indicates which data values are not formatted as a missing data type (na)
# pandas.DataFrame.fillna(df, value=20) – Fill-in any missing values in the df with the integer 20.

# Feature engineering
# pandas.get_dummies(data) – One-hot encode categorical data and automatically create a data frame of those encoded values in dummy columns. 


