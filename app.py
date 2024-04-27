import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from keras.models import load_model,Sequential
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM

model = load_model(r"C:\Users\ASUS\OneDrive\Desktop\Minor Project\Stock_Price_Analysis\Stock Price Prediction Model.keras")


# App title
st.markdown('''
# Stock Price Analysis
''')
st.write('---')


# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2013, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2024, 4, 10))

# Retrieving tickers data
ticker_list = pd.read_csv("data.csv")
tickerSymbol = st.sidebar.selectbox('Stock Symbol', ticker_list['Symbol'] +" - " + ticker_list['Name']) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol.split(" ")[0]) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
df = yf.download(tickerSymbol.split(" ")[0], start_date ,end_date)

def remove_prefix(url,pre):
    if url.startswith(pre):
        return url[len(pre):]
    return url

prefix = "http://"
prefix2  = "http://www."

link = tickerData.info['website']
logo_link = remove_prefix(link,prefix2);
if link == logo_link:
    logo_link = remove_prefix(link,prefix)

logo_name = f"https://logo.clearbit.com/{logo_link}"
string_logo = '<img src=%s>' % logo_name
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName'] + " (" + tickerSymbol.split(" ")[0] + ")"
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Stock Data**')
st.write(df.style.set_table_attributes("style='width: 700px; height: 400px;'"))

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

####
# st.write(tickerData.info)

# data = yf.download(tickerSymbol, start_date ,end_date)
#Visualize the closing price history
st.subheader('Closing Price History')
fig1 = plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.plot(df['Close'])
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price USD ($)",fontsize=18)
plt.legend(['Closing Price'],loc = "lower right")
plt.show()
st.pyplot(fig1)

#Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = df.Close.rolling(50).mean()
fig2 = plt.figure(figsize=(16,8))
plt.title('Price vs MA50')
plt.plot(df.Close, 'g')
plt.plot(ma_50_days, 'r')
plt.legend(['Actual Value','Moving Average 50 Days'],loc = 'upper left')
plt.show()
st.pyplot(fig2)

#Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
fig3 = plt.figure(figsize=(16,8))
plt.title('Price vs MA50 vs MA100')
plt.plot(df.Close, 'g')
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.legend(['Actual Value','Moving Average 50 Days','Moving Average 100 Days'],loc = 'upper left')
plt.show()
st.pyplot(fig3)

#Create a new dataframe with only the Close column
data = df.filter(['Close'])
#Convert a dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8);
# training_data_len

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the testing dataset
#Create a new array containing scaled values
test_data = scaled_data[training_data_len - 100: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]

for i in range(100,len(test_data)):
  x_test.append(test_data[i-100:i,0])

#Convert the data into a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#Get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
# rmse

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, predictions)

print("Mean Absolute Percentage Error (MAPE):", mape)

# Plot the true prices and predicted prices
st.subheader('True vs Predicted Stock Prices')
fig5 = plt.figure(figsize=(16,8))
plt.plot(y_test, label='True Prices', marker='o')
plt.plot(predictions, label='Predicted Prices', marker='x')
plt.xlabel('Days',fontsize=18)
plt.ylabel('Price',fontsize=18)
plt.title('True vs Predicted Stock Prices')
plt.legend()
plt.show()
st.pyplot(fig5)

# Plot the Mean Absolute Percentage Error (MAPE)
st.subheader('Mean Absolute Percentage Error (MAPE)')
fig6 = plt.figure(figsize=(16,8))
plt.plot(np.arange(1, len(y_test) + 1), np.abs((np.array(y_test) - np.array(predictions)) / np.array(y_test)) * 100, label='MAPE', marker='o', color='green')
plt.xlabel('Days',fontsize=18)
plt.ylabel('MAPE (%)',fontsize=18)
plt.title('Mean Absolute Percentage Error (MAPE)')
plt.legend()
plt.show()
st.pyplot(fig6)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
st.subheader('Prediction using LSTM Model')
fig7 = plt.figure(figsize=(16,8))
plt.title("LSTM Model")
plt.xlabel("Data",fontsize=18)
plt.ylabel("Close Price USD ($)",fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close",'Predictions']])
plt.legend(['Train','True Value','Predictions'],loc = 'lower right')
plt.show()
st.pyplot(fig7)

#Get the quote
start = '2012-01-01'
end = datetime.datetime.now().date()
stock = 'AAPL'
df1 = yf.download(stock, start, end)
new_df = df1.filter(["Close"])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#Get the preidcted scaled price
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
# print(pred_price)
pred_price = pred_price[0][0]
st.sidebar.write(f"Predicted Stock Price : $ %.6f " % pred_price)


st.write('---')

st.write("""**Credits**
 - App built by Sneh Kumar Daluka (21U02068) and Vineet Patel (21U02020)""")
