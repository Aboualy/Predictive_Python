import pandas as pd
import numpy as np
import datetime as dt
from __future__ import absolute_import

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader.data import DataReader
import matplotlib.pyplot as plt
from numpy import loadtxt, where


end = dt.datetime.now()
start = end - dt.timedelta(days=5 * 356)

df = DataReader("MU", "iex", start, end)

# df.reset_index(inplace = True)

df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df['year'] = df.index.year.values
df['month'] = df.index.month.values
df['day'] = df.index.day.values

df.head()

# print(year)
df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)

# df['date'] = pd.to_datetime(df[['year', 'month','day']])
df.date.values

df.tail()

df.close.plot(figsize=(12, 8), title='MU')

# for i, (index, row) in enumerate(df.iterrows()):
# print (row)
# print( df.loc[df.index[ i - 4 ], 'close'])
# df.at[index,'Momentum_function']

# Selected technical indicators and their formulas (Type 1).
# Stochastic %K
lowest = df['low'].rolling(window=4).min()
df['Stochastic_k'] = pd.Series((df['close'] - lowest)) / (df['high'] - lowest)


# Calculating in two different ways
def count_momentum(dataFrame, periods, column_close='close'):
    dataFrame['Momentum'] = np.nan
    for i, (index, row) in enumerate(df.iterrows()):
        if i >= periods:
            previous_value = df.loc[df.index[i - periods], column_close]
            current_value = df.loc[df.index[i], column_close]
            final_value = (current_value - previous_value)
            dataFrame.at[index, 'Momentum'] = final_value

    return dataFrame


df['Momentum_function'] = df.close - df.close.shift(4)


# df['Momentum_Test'] = pd.Series(df['close'].diff(4))

# ROC (rate of change)	Ct/C(t−n) × 100,
def ROC(dataFrame, periods, column_close='close'):
    dataFrame['ROC'] = np.nan
    for i, (index, row) in enumerate(df.iterrows()):
        if i >= periods:
            momentum = dataFrame.at[index, 'Momentum']
            shift_value = df.loc[df.index[i - periods], column_close]
            ROC = momentum / shift_value
            dataFrame.at[index, 'ROC'] = ROC

    return dataFrame


df['ROC_function'] = pd.Series(df['close'].diff(4) / df['close'].shift(4))
# df['ROC_Test2'] = df.close.pct_change(4)

count_momentum(df, 4)
ROC(df, 4)
df.head()

# count_ROC(df,5)
# def count_ROC(dataFrame, numberOfDays):
# dataFrame['ROC_Test1'] = pd.Series(dataFrame['close'].diff(numberOfDays - 1) /
# dataFrame['close'].shift(numberOfDays - 1), )
# return dataFrame.ROC_Test1
# https://www.youtube.com/watch?v=4AMGMWQosps
# previous_value = df.loc[df.index[i-periods], column_close]
# current_value = df.loc[df.index[i], column_close]
# momentum = (current_value - previous_value)

df[['open', 'close', 'high', 'low']].plot(figsize=(18, 8), title='MU')

# Step 2 - implement 2 features and visualize the price and the features in the same graph
df[['ROC', 'Momentum', 'close']].plot(figsize=(18, 8), title='Features')

# Step 5 - add 2 more features from the “Type 2” category of features presented in the paper
# Selected technical indicators and their formulas (Type 2).

# BIAS6
MA6 = df['close'].rolling(6, min_periods=4).mean()
df['BIAS6'] = pd.Series((df['close'] - MA6) / MA6)


# https://www.investopedia.com/terms/o/onbalancevolume.asp
def count_OBV(dataFrame):
    column_volume = 'volume'
    column_close = 'close'
    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            previous_OBV = dataFrame.loc[dataFrame.index[i - 1], 'OBV']
            if row[column_close] > dataFrame.loc[dataFrame.index[i - 1], column_close]:
                current_OBV = previous_OBV + row[column_volume]
            elif row[column_close] < dataFrame.loc[dataFrame.index[i - 1], column_close]:
                current_OBV = previous_OBV - row[column_volume]
            else:
                current_OBV = previous_OBV
        else:
            previous_OBV = 0
            current_OBV = row[column_volume]

        dataFrame.at[index, 'OBV'] = current_OBV

    return dataFrame


count_OBV(df)


# MA5	MA5=(∑5i=1Ct−i+1)/5
def count_moving_average(dataFrame, periods, column_close='close'):
    dataFrame['MA_function'] = np.nan
    for i, (index, row) in enumerate(df.iterrows()):
        if i >= periods - 1:
            sum = 0
            for j in range(periods):
                sum += df.loc[df.index[i - j], column_close]
            ma_value = sum / periods
            dataFrame.at[index, 'MA_function'] = ma_value

    return dataFrame


count_moving_average(df, 5)

df['Moving_Average_5'] = pd.Series(df['close'].rolling(5, min_periods=5).mean())
df.head(10)

df[['MA_function', 'OBV', 'close']].plot(figsize=(18, 8), title='MU')

df[['ROC', 'Momentum', 'MA_function', 'OBV', 'close']].plot(figsize=(18, 8), title='Features')

df['label'] = df.close.shift(-1)
df = df.dropna()

df.head()

# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model

X = np.array(df[['ROC', 'Momentum', 'close']])
Y = np.array(df.label)
Y = Y.reshape(-1, 1)

# X = np.nan_to_num(X)
# Y = np.nan_to_num(Y)
# scaler = preprocessing.StandardScaler().fit(Y)
# X = scaler.transform(X)
# Y = scaler.transform(Y)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import datetime

# train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# %time df['date'] = pd.to_datetime(df.index)


tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    date_train, date_test = df.date[train_index], df.date[test_index]

# Use a linearmodel, seelinkfor moremodelsto test

linear = ElasticNetCV()
linear.fit(X_train, y_train.reshape(len(y_train), ))

# A simple score measure•
# Run your regression forecast
# Store your forecast values in an array
# y_test= np.nan_to_num(y_test)
linear.score(X_test, y_test)

print('Variance score: %.2f' % linear.score(X_test, y_test))

from sklearn.metrics import mean_squared_error

forcast_set = linear.predict(X_test)
# The coefficients
print('Coefficients:', linear.coef_)

# The mean square error
print("Residual sum of squares: %.2f" % np.mean((linear.predict(X_test) - y_test) ** 2))

print('Mean squared error: ', mean_squared_error(forcast_set, y_test))

# Plot outputs
from pylab import *

# fig, ax = plt.subplots()
# ax.set_ylim(0,y_test.max())
# ax.set_xlim(0,y_test.max())

plt.scatter(forcast_set, y_test, color='g')
# plt.plot(np.sort(X_test[:,0], axis=0), forcast_set, color='black', linewidth=3)
# plt.plot(np.sort(X_test, axis=0), forcast_set, color='red', linewidth=3)
plt.plot(X_test, forcast_set, color='red', linewidth=3)
plt.show()

dataFrame = pd.DataFrame({'Actual': y_test.ravel(), 'Predicted': forcast_set, 'date': date_test})
# %time dataFrame ['date'] = pd.to_datetime(date_test.values)

# Plot outputs
from pylab import *

# fig, ax = plt.subplots()
# ax.set_ylim(0,y_test.max())
# ax.set_xlim(0,y_test.max())
# print(date_test.values)
# plt.plot(np.sort(X_test[:,0], axis=0), forcast_set, color='black', linewidth=3)
# date_test = matplotlib.dates.date2num(date_test)

plt.scatter(forcast_set, y_test, color='g')

# %time data [date = pd.DataFrame (pd.to_datetime(date_test))

plt.plot(dataFrame.date, dataFrame.Predicted, color='red', linewidth=3)

plt.show()

# view limit minimum -36806.770608854764 is less than 1 and is an invalid Matplotlib date value. This often happens if you pass a non-datetime value to an axis that has datetime units

forcast_set[:5]

y_train[:5]

# Step 5 - adding 2 more features from the “Type 2” category of features presented in the paper

y_test[:5]

print(y_test.ravel()[:5])

dataFrame.head()

# Just a test
for index, row in dataFrame.iterrows():
    if dataFrame.at[index, 'Actual'] <= 0:
        print(dataFrame.at[index, 'Actual'])
        print(index)
    elif dataFrame.at[index, 'Predicted'] <= 0:
        print(dataFrame.at[index, 'Predicted'])
        print(index)


def make_decision(dataFrame, column_actaul='Actual', column_predicted='Predicted'):
    Predicted_sub = []
    Predicted_UP_DOWN = []
    Actual_UP_DOWN = []
    Dec = []

    for index, row in dataFrame.iterrows():
        Predicted_Actual = dataFrame.at[index, 'Predicted'] - dataFrame.at[index, 'Actual']
        Predicted_sub.append(Predicted_Actual)

        # Actual_Output = dataFrame.at[index, 'Actual'] - dataFrame.at[index, 'Predicted']
        # Actual_sub.append (Actual_Output)

        if Predicted_Actual > 0:
            Predicted_UP_DOWN.append("UP")
            Actual_UP_DOWN.append("DOWN")
            Dec.append("Buy")

        elif Predicted_Actual < 0:
            Predicted_UP_DOWN.append("DOWN")
            Actual_UP_DOWN.append("UP")
            Dec.append("Sell")
        else:
            Predicted_UP_DOWN.append("STABLE")
            Actual_UP_DOWN.append("STABLE")
            Dec.append("Sell")

    dataFrame['Predicted_UP_DOWN'] = Predicted_UP_DOWN
    dataFrame['Actual_UP_DOWN'] = Actual_UP_DOWN
    dataFrame['Decision'] = Dec
    dataFrame['Predicted-Actual'] = Predicted_sub
    return dataFrame


make_decision(dataFrame)
dataFrame[['Actual', 'Predicted', 'Predicted-Actual', 'date']].plot(figsize=(18, 8), title='Decision')
dataFrame.head()

dataFrame[['Actual_UP_DOWN', 'Predicted_UP_DOWN', 'Decision']].apply(pd.value_counts).plot(kind='bar', figsize=(18, 12),
                                                                                           subplots=True)

# %time dataFrame['ts'] = pd.to_datetime(dataFrame.date)
# dataFrame.apply(pd.value_counts).plot(kind='bar', subplots=True)
# dataFrame[['Actual_UP_DOWN','Predicted_UP_DOWN','Decision']].plot(x=dataFrame.index.values, kind="bar")




def make_decision(dataFrame, column_label='label', column_close='close'):
    decision = []
    for i, (index, row) in enumerate(df.iterrows()):
        if row[column_label] > dataFrame.loc[dataFrame.index[i - 1], column_close]:
            # print( dataFrame.loc[dataFrame.index[i-1], column_close])
            # print(i)
            decision.append("Buy")
        elif row[column_label] < dataFrame.loc[dataFrame.index[i - 1], column_close]:
            decision.append("Sell")
            # print( dataFrame.loc[dataFrame.index[i-1], column_close])
            # print(i)
        else:
            decision.append("Sell")

    dataFrame['Decision'] = decision
    return dataFrame
# df.Decision.value_counts().plot('bar', figsize=(18,8), title='Decision')