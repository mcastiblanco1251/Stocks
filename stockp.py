import math
import time
import pandas_datareader.data as pdr 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import date
import streamlit as st
#from pandas_datareader import data as pdr
import yfinance as yf
from datetime import date
from PIL import Image
import smtplib

im = Image.open("bolsa.jpg")
st.set_page_config(page_title='DeepLearningLSTM', layout="wide", page_icon=im)
#st.set_option('deprecation.showPyplotGlobalUse', False)
#, caption='Sunrise by the mountains')

# st.write("""
# # Stock Prediction App
# This app predicts the Stocks!
# """)
row1_1, row1_2 = st.columns((2,3))

with row1_1:
    image = Image.open('stock.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Stock Prediction App
    This app use Deep Learning LSTM to predict!
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name=st.text_input('Name')
            mail = st.text_input('Email')
            q=st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com",587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n'+name + '\n'+mail+'\n'+ q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Application')
st.write('_______________________________________________________________________________________________________')
app_des=st.expander('Description App')
with app_des:
    st.markdown("""
    This app is based in deep learning LSTM model alrgorith to predict stocks. Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

    LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

    All recurrent neural networks have the form of a chain of repeating modules of neural network.
        """)



st.sidebar.header('Inputs User')
uploaded_file = st.sidebar.file_uploader("Upload file CSV ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        end= st.sidebar.date_input("Final Date")
        #yf.pdr_override()
        #d=date.today()
        s_list=pd.read_csv('stocks.csv')
        stock= st.sidebar.selectbox('Select Symbol Stock',(s_list.Symbol))
        n=s_list[s_list['Symbol']==stock].index.item()
        name=s_list['Name'][n]
        st.sidebar.text_input('Company Name', name)

#st.subheader(name)
#data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
        df=yf.download(stock, start='2012-01-01', end=end)
        return name, df, end
    df=user_input_features()
df2=df[1]
name=df[0]
end=df[2]

st.header('Representation')

st.subheader('Company Name: '+ name )

row2_1, row2_2, = st.columns((2,2))

with row2_1:
    st.subheader('Stock Graphic Currency')
    fig=plt.figure(figsize=(12,10))
    plt.title('Close Price History')
    plt.plot(df2['Close'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    st.pyplot(fig)

with row2_2:
    st.subheader('Stock Table Currency')
    st.write(df2.tail(12))


#Create a new dataframe with only the 'Close' column
data = df2.filter(['Close'])
st.write(data)
#data

#Converting the dataframe to a numpy array
dataset = data.values
#dataset

#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
#training_data_len

#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler

scaled_data = scaler.fit_transform(dataset)
#scaled_data

#Create the scaled training data set
train_data = scaled_data[0:training_data_len, : ]
#train_data.shape

#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#x_train.shape, y_train.shape

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#x_train

#Build the LSTM network model
model = Sequential()

model.add(LSTM(units=25, return_sequences=True,input_shape=(x_train.shape[1],1)))
#model.add(Dropout(0.2))
model.add(LSTM(units=25, return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(units=25))
#model.add(Dropout(0.2))
model.add(Dense(units=1))

#Compile the model

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

#st.subheader('Model')
#st.write(str(model.summary()))

#Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10)

#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#test_data

#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Convert x_test to a numpy array
x_test = np.array(x_test)
#x_test

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


#Getting the models predicted price values
predictions = model.predict(x_test)
scaler=scaler.fit(dataset)
predictions = scaler.inverse_transform(predictions)#Undo scaling
#predictions

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#rmse

import sklearn
from sklearn.metrics import r2_score
r2=sklearn.metrics.r2_score(predictions, y_test)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(predictions, y_test)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(predictions, y_test)


# latest_iteration = st.empty()
# bar = st.progress(0)
# iter=50
# for i in range(iter):
#     latest_iteration.text(f'Progress {i*(150//iter)}%')
#     bar.progress(i *(150//iter))
#     time.sleep(0.1)


st.subheader('Acurracy Evaluation')
#acc={'RMSE': rmse, 'R2':r2, 'MSE':mse, 'MAE':mae}
st.write(pd.DataFrame({'RMSE': rmse, 'R2':r2, 'MSE':mse,'MAE':mae}, columns=['RMSE', 'R2', 'MSE', 'MAE'], index=['Acurracy']))


if r2>=0.90:
    st.write('**Great Performance!!!**')
elif r2>=0.8 and r2<0.9:
    st.write('**Aceptable Performance**')
elif r2>0.6 and r2<0.8:
    st.write('**Regular Performance**')
elif r2<0.6:
    st.write('**Bad Performance**')

# training metrics
#scores = model.evaluate(x_train, y_train, verbose=1, batch_size=200)
#print('Accurracy: {}'.format(scores[1]))

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#valid

st.header('Prediction')
st.subheader('Company name: ' + name)
row3_1, row3_2, = st.columns((2,2))

with row3_1:
#Visualize the data
    st.subheader('Graphic Real an Predictions Prices')
    fig=plt.figure(figsize=(12,10))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
    #plt.show()
    st.pyplot(fig)
#Show the valid and predicted prices

with row3_2:
    st.subheader('Real and Predictes Prices')
    st.write(valid.tail(12))

#Get the quote
stock_quote = df2#pdr.get_data_yahoo(stock, start='2012-01-01', end=end)
#Create a new dataframe
new_df = stock_quote.filter(['Close'])
#Get teh last 60 day closing price
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
string = ' '.join(str(x) for x in pred_price)
st.subheader(f'Future Price date {end} is USD${string[1:9]}')
#st.sidebar.text_input('Future Price:', str(pred_price))

#Contact Form

with st.expander('Help? 👉'):
    st.markdown(
            " App Help? contact to [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)")
