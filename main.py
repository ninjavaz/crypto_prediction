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
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas_datareader.data as web
plt.style.use('dark_background')





class CurrencyData:
    def __init__(self, currency):
        
        self.currency = currency
    
    
    
    
    

    

    def generate_full_history(self):
        global df
        global close_set
        global close
        #set start and end of the market chart
        end = datetime.today()
        #start = datetime(end.year-15,end.month,end.day)
        start=datetime(2000, 1, 1) #max time period
        
        #downloading market chart from start to end for XRP-USD
        df = web.DataReader(self.currency, 'yahoo', start, end)
        
        #New dataframe with only Close column
        close = df.filter(["Close"]) #choose only one column
        close_set = close.values
        



    def vis_data(self, data):#visualise the closing price history of XRP
        title = "Closing Price History of %s" % self.currency[0:3]
        xlabel = "Date"
        ylabel =  "Closing price [%s]" % self.currency[4:6]
        
        plt.figure(figsize=(16,8))
        plt.title(title)
        plt.plot(data)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.show()


    def make_predict(self, start):
        global close_test
        global close_train
        start1 = np.datetime64(start) #converting string like 'YYYY-MM-DD' to datetime format
        data_test = web.DataReader(self.currency, 'yahoo', start1, datetime.today())
        close_test = data_test.filter(["Close"])
        
        data_train = web.DataReader(self.currency,'yahoo', datetime(2000, 1, 1), start1 ) #Create data for training from 0 to start1, which is start of close_test
        close_train = data_train.filter(["Close"])
        
        print(close_test)
        print(close_train)


    
        

        

    def create_train_data(self):
        global x_train, y_train
        #Scaling data
        scaler = MinMaxScaler(feature_range=(0,1))

        #Create the training data set
        scaled_close_train = scaler.fit_transform(close_train)
        print(scaled_close_train)
        
        #Split the data into x and y _train sets
        x_train = []
        y_train = []

        for i in range(60, len(scaled_close_train)):
            x_train.append(scaled_close_train[i-60:i, 0])
            y_train.append(scaled_close_train[i, 0])
            if i<=60:
                print(x_train)
                print(y_train)
                print()
                
    def convert_xy(self):
        global x_train, y_train
        
        #Converting the x,y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        #Reshape the data because LSTM needs 3-dimensional data set
        x_train =np.reshape (x_train, (x_train.shape[0], x_train.shape[1], 1))

        
    def lstm_model(self):
        #building lstm neural model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True))



         

        
    

xrp = CurrencyData('XRP-USD') #currency must be 'ABC-DEF' where ABC - crypto currency and DEF - fiat currency
  
xrp.generate_full_history()
xrp.make_predict('2017-02-25') #taking selected interval from start to today of data from with header Close instead of random typing testing interval
xrp.vis_data(close) #vis_data(dataframe)
xrp.create_train_data()
xrp.convert_xy()
xrp.lstm_model()














