# -*- coding: utf-8 -*-




#I used RNN in this program - Long Short Term Memory(LSTM)
#to predict price of selected cryptocurrency in specified peroid of time and then check it
#with reality

from datetime import datetime
import math
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

#set start and end of the market chart
end = datetime.today()
start = datetime(end.year-5,end.month,end.day)
#downloading market chart from start to end for XRP-USD
df = web.DataReader('XRP-USD', 'yahoo', start, end)







#downloading actual price of USD to PLN from NBP site
url = 'http://api.nbp.pl/api/exchangerates/rates/A/USD/' #API NBP
res = requests.get(url)
html_page = str(res.content)

#Converting data to obtain pure float number
index_of_mid = html_page.index("mid")

usd_price = float(html_page[index_of_mid+5:index_of_mid+11])

df[["Adj Close", "High", "Low", "Open", "Close", "Volume"]] *= usd_price #convert USD to PLN


#visualise the closing price history of XRP

plt.figure(figsize=(16,8))
plt.title("Closing Price History")
plt.plot(df['Close'])

plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price Size PLN zł", fontsize=18)

plt.show()




