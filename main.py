import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import datetime
from statsmodels.tsa.stattools import adfuller
data = pd.read_excel("Analyst Test Questions_v2.xlsx",sheet_name=1, index_col=1)
data= data.sort_index()
list(data.columns.values)[1]
t=0
datetimes = []
values = []
for idx, date in enumerate(data.index):
    k=1
    for i in data.iloc[idx,1:-1]:
        header = data.columns.values[k]
        header=datetime.datetime.combine(date,header)
        #print(i)
        datetimes.append(header)
        values.append(i)
        #plt.plot(t, i, linestyle = 'dotted')
        k+=1
        t+=1



formatteddata = pd.DataFrame(data={'DateTimes':datetimes , 'Values':values}).sort_values('DateTimes')
formatteddata = formatteddata.reset_index(drop=True)
#formatteddata.index= formatteddata('DateTimes')
plt.figure(figsize=(30,5))
plt.plot(datetimes,values)
plt.xlabel('Date & Time')
plt.ylabel('Half hourly usage')
#plt.figure(figsize=(20,5))
plt.show()


import matplotlib
means=[]
times = []
for idx, time in enumerate(data):
    if idx == 0 or time == 'Total':
        continue
    times.append(datetime.datetime.combine(date,time))
    means.append(data[time].mean()) 

dailymeanframe = pd.DataFrame(data={'Times':times , 'Means':means}).sort_values('Times')
plt.figure(figsize=(20,5))
plt.plot(dailymeanframe['Times'],dailymeanframe['Means'])
plt.xlabel("Time of Day")
plt.ylabel("Half hour kWh")
xformatter = matplotlib.dates.DateFormatter('%H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
plt.show()


means=[]
times = []
for idx, time in enumerate(data):
    if idx == 0 or time == 'Total':
        continue
    times.append(datetime.datetime.combine(date,time))
    means.append(data[time].max()) 

dailymeanframe = pd.DataFrame(data={'Times':times , 'Means':means}).sort_values('Times')
plt.figure(figsize=(20,5))
plt.plot(dailymeanframe['Times'],dailymeanframe['Means'])
plt.show()





#plt.plot(df['date_column'], df['value_column'], linestyle = 'dotted')



totals = pd.DataFrame(data={'Dates':[x for x in data.index], 'Total': [x for x in data['Total']]})
data2 = pd.read_excel("Analyst Test Questions_v2.xlsx",sheet_name=2)
regularised = []
for idx, value in enumerate(formatteddata.iloc):
    #print(idx)
    for i, entry in enumerate(data2.iloc):
        if value['DateTimes'].month == entry['Date'].month and value['DateTimes'].year == entry['Date'].year:
            regularised.append(value['Values'] / entry['HDD at 15.5°C'])

formatteddata['Regularised'] = regularised
regularised = []
for idx, value in enumerate(totals.iloc):
    #print(idx)
    for i, entry in enumerate(data2.iloc):
        if value['Dates'].month == entry['Date'].month and value['Dates'].year == entry['Date'].year:
            regularised.append(value['Total'] / entry['HDD at 15.5°C'])


plt.plot(totals['Dates'],totals['Total'])
plt.xlabel("Date")
plt.ylabel("Total Usage per day")
plt.show()


totals['Regularised'] = regularised

plt.plot(totals['Dates'],totals['Regularised'])
plt.xlabel("Date")
plt.ylabel("Normalised total Usage per day")
plt.show()



plt.plot(formatteddata['DateTimes'],formatteddata['Regularised'])
plt.show()

train,test = formatteddata[:int(len(formatteddata)*0.9)] , formatteddata[int(len(formatteddata)*0.9):]

plt.plot(train['DateTimes'],train['Regularised'],'green')
plt.plot(test['DateTimes'],test['Regularised'],'blue')
plt.show()

# SARIMAXmodel = SARIMAX(train['Regularised'], order = (1, 0, 1), seasonal_order=(2,0,2,12),initialization='approximate_diffuse')
# SARIMAXmodel = SARIMAXmodel.fit()
# y_pred = SARIMAXmodel.get_forecast(len(test.index))
# y_pred_df = y_pred.conf_int(alpha = 0.05) 
# test["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])



# plt.plot(test['Dates'],test['Regularised'],'green')
# plt.plot(test['Dates'],test['Predictions'],'blue')
# plt.show()


decomposition = sm.tsa.seasonal_decompose(formatteddata['Regularised'], 
                                          model='additive', 
                                          period=48)

# Extract the decomposed components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
decomposition.plot()
plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(formatteddata['Regularised'])

# Extract and print the test statistics and p-value
test_statistic = result[0]
p_value = result[1]
print(f"Test Statistic: {test_statistic}")
print(f"P-value: {p_value}")

df_temp_max = formatteddata['Regularised']


# Plot ACF
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(df_temp_max, ax=ax)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(df_temp_max, ax=ax)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Split into training and testing
df_train = train['Regularised']
df_test = test['Regularised']

# Plot the last 10 years of training data and the 2 of testing
ax = df_train[-12*10:].plot(figsize=(10, 5))
df_test.plot(ax=ax)
plt.legend(['Train', 'Test'])
plt.xlabel('Date')
plt.ylabel('Maximum temperature')
plt.show()


import itertools
import math

# Define the range of values for p, d, q, P, D, Q, and m
p_values = range(0, 10)  # Autoregressive order
d_values = range(0,10)          # Differencing order
q_values = range(0,10)  # Moving average order
P_values = range(0, 10)  # Seasonal autoregressive order
D_values = range(0, 10)  # Seasonal differencing order
Q_values = range(0, 10)  # Seasonal moving average order
m_values = [48]         # Seasonal period

# Create all possible combinations of SARIMA parameters
param_combinations = list(itertools.product(p_values, 
                                            d_values, 
                                            q_values, 
                                            P_values, 
                                            D_values, 
                                            Q_values, 
                                            m_values))

# Initialize AIC with a large value
best_aic = float("inf")  
best_params = None

# Perform grid search
for params in param_combinations:
    order = params[:3]
    seasonal_order = params[3:]
    
    try:
        model = sm.tsa.SARIMAX(df_train, 
                               order=order, 
                               easonal_order=seasonal_order)
        result = model.fit(disp=False)
        aic = result.aic
        
        # Ensure the convergence of the model
        if not math.isinf(result.zvalues.mean()):
            print(order, seasonal_order, aic)
        
            if aic < best_aic:
                best_aic = aic
                best_params = params
                
        else:
            print(order, seasonal_order, 'not converged')

    except:
        continue

# Print the best parameters and AIC
print("Best Parameters:", best_params)
print("Best AIC:", best_aic)

model = sm.tsa.SARIMAX(df_train,
                       order=best_params[:3],
                       seasonal_order=best_params[3:])
result = model.fit(disp=False)

# Show the summary
result.summary()

forecast = result.get_forecast(steps=17520)
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

ax = df_train[-12*4:].plot(figsize=(10,5))
forecast_values.plot()
df_test.plot(ax=ax)
plt.fill_between(forecast_values.index, 
                 confidence_intervals['lower Regularised'], 
                 confidence_intervals['upper Regularised'], 
                 color='blue',
                 alpha=0.15)
plt.legend(['Training max temp', 
            'Forecast max temp', 
            'Actual max temp'], 
           loc='upper left')
plt.xlabel('Date')
plt.ylabel('Maximum temperature')
plt.grid(alpha=0.5)
plt.show()



df_train,df_test = totals[:int(len(totals)*0.9)]['Regularised']*1000 , totals[int(len(totals)*0.9):]['Regularised']*1000

# Import the library
from pmdarima.arima import auto_arima
print('fitting')
# Build and fit the AutoARIMA model
model = auto_arima(df_train,  
                   suppress_warnings=False)
result = model.fit(df_train)


result.summary()

forecast_auto, conf_int_auto = model.predict(n_periods=30,
                                             return_conf_int=True)

# Get forecast and confidence intervals for two years
forecast_values_auto = forecast_auto
confidence_intervals_auto = conf_int_auto

# Plot forecast with training data
ax = df_train[-12*4:].plot(figsize=(10,5))
forecast_auto.plot(ax=ax)
df_test.plot(ax=ax)
plt.fill_between(forecast_values_auto.index, 
                 confidence_intervals_auto[:,[0]].flatten(), 
                 confidence_intervals_auto[:,[1]].flatten(), 
                 color='blue',
                 alpha=0.15)
plt.legend(['Training max temp', 
            'Forecast max temp', 
            'Actual max temp'], 
           loc='upper left')
plt.xlabel('Date')
plt.ylabel('Maximum temperature')
plt.grid(alpha=0.5)
plt.show()


from statsmodels.tsa.api import ExponentialSmoothing
es = ExponentialSmoothing(train['Values']).fit()

plt.plot(train['Values'],'blue')
plt.plot(es.predict(0, 17000),'green')
plt.show()
