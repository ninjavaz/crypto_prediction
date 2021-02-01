# -*- coding: utf-8 -*-




#I used RNN in this program - Long Short Term Memory(LSTM)
#to predict price of selected cryptocurrency in specified peroid of time and then check it
#with reality

from datetime import datetime
import math
from numpy.core.numeric import NaN
from numpy.ma.core import maximum_fill_value
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
import pandas_datareader.data as web
plt.style.use('dark_background')
import json
import requests

#usd_price = 0

global df #Data frame of currency price



def change_PLN(dataframe): #function that change USD to PLN in dataframe
    url = 'http://api.nbp.pl/api/exchangerates/rates/A/USD/' #API NBP
    res = requests.get(url)
    html_page = str(res.content)
    #Converting data to obtain pure float number
    index_of_mid = html_page.index("mid")
    usd_price = float(html_page[index_of_mid+5:index_of_mid+11])
    dataframe *= usd_price




def generate_full_history():
    global df
    #set start and end of the market chart
    end = datetime.today()
    #start = datetime(end.year-15,end.month,end.day)
    start=datetime(2000, 1, 1) #max time period
    #downloading market chart from start to end for XRP-USD
    df = web.DataReader('XRP-USD', 'yahoo', start, end)
    #downloading actual price of USD to PLN from NBP site
    change_PLN(df[["Adj Close", "High", "Low", "Open", "Close", "Volume"]])
    



generate_full_history() #generating full history of XRP closing prices in PLN


def vis_data(title, data, xlabel, ylabel):#visualise the closing price history of XRP
    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.plot(data)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.show()

def make_predict(start, end):
       
    start1 = np.datetime64(start) #converting string like 'YYYY-MM-DD' to datetime format
    end1 = np.datetime64(end)
    data_test = web.DataReader('XRP-USD', 'yahoo', start1, end1)
    close_test = data_test.filter(["Close"])
    change_PLN(close_test)
    print(close_test)



vis_data("Closing Price History of XRP", df["Close"], "Date", "Closing price [PLN]" ) #vis_data(title, data, xlabel, ylabel)
    
make_predict('2015-02-25','2021-01-31') #taking selected interval of data from with header Close and changing it to PLN values, instead of random typing testing interval






# #New dataframe with only Close column
# close = df.filter(["Close"]) #choose only one column
# close_set = close.values





