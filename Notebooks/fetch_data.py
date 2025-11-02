#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass



# Data Exploration - To get specific ticker data

# In[38]:


start_date = datetime.datetime(2015, 1, 1).date()
end_date = datetime.datetime.now().date()
start_date, end_date


# In[39]:


tickers = "NVDA"


# In[40]:


nvda = yf.Ticker(tickers)


# In[41]:


historical_data = nvda.history(start = start_date, end = end_date, interval = '1d')


# In[42]:


historical_data.head()


# In[43]:


historical_data.describe()


# In[44]:


fig = plt.figure()

plt.plot(historical_data.Close)

plt.legend(["Close", "Open"])


# Feature Engineering - Dataset

# In[45]:


historical_data.drop(columns=["Dividends", "Stock Splits", "Volume"], inplace=True)


# In[46]:


historical_data.head()


# In[47]:


print(historical_data.columns)


# In[48]:


present_date = historical_data.index.max()
weekday = present_date.isoweekday()
days_to_add = 1 if weekday not in [5, 6] else (8 - weekday)
next_date = present_date + pd.Timedelta(days=days_to_add)

print(f"Present date: {present_date}")
print(f"Next valid date: {next_date}")

test_row = pd.DataFrame({'Date': [next_date],**{col: [0.0] for col in historical_data.columns if col != 'Date'}})
test_row.head()


# Since the useful columns are - Date, Close(shows closing price), We need an adj_Close (for splits/dividends).
# We can drop the rest

# We need Lag Features for each day, to keep track of last traded price.

# In[49]:


for i in range(1, 7):
    historical_data[f"Close_lag_{i}"] = historical_data.Close.shift(periods=i, axis=0)
    historical_data[f"Open_lag_{i}"] = historical_data.Open.shift(periods=i, axis=0)
    historical_data[f"High_lag_{i}"] = historical_data.High.shift(periods=i, axis=0)
    historical_data[f"Low_lag_{i}"] = historical_data.Low.shift(periods=i, axis=0)

historical_data.head()


# In[50]:


historical_data.drop(columns = ["Open","High","Low"],inplace = True)


# In[51]:


historical_data.fillna(0, inplace = True)
historical_data.head()


# Defining a function to do this 

# In[56]:


historical_data.reset_index(inplace=True)
print(historical_data.columns)


# In[54]:


def get_stock_data(ticker: str):        
    start_date = datetime.datetime(2015, 1, 1).date()
    end_date = datetime.datetime.now().date()
    try:
        check = yf.Ticker(ticker)
    except:
        print("Error in fetching data")
        return
    historical_data = check.history(start = start_date, end = end_date, interval = '1d')
    historical_data.drop(columns=["Dividends", "Stock Splits", "Volume"], inplace=True)
    present_date = historical_data.index.max()
    weekday = present_date.isoweekday()
    days_to_add = 1 if weekday not in [5, 6] else (8 - weekday)
    next_date = present_date + pd.Timedelta(days=days_to_add)
    test_row = pd.DataFrame({'Date': [next_date],**{col: [0.0] for col in historical_data.columns if col != 'Date'}})
    for i in range(1, 7):
        historical_data[f"Close_lag_{i}"] = historical_data.Close.shift(periods=i, axis=0)
        historical_data[f"Open_lag_{i}"] = historical_data.Open.shift(periods=i, axis=0)
        historical_data[f"High_lag_{i}"] = historical_data.High.shift(periods=i, axis=0)
        historical_data[f"Low_lag_{i}"] = historical_data.Low.shift(periods=i, axis=0)
    historical_data.drop(columns = ["Open","High","Low"],inplace = True)
    historical_data.fillna(0, inplace = True)
    historical_data.reset_index(inplace=True)
    return historical_data

