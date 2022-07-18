import streamlit as st
from datetime import date
#import yfinance as yf
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import requests

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Cryptocurrencies Prediction App')
crypto_currencies=('ethereum', 'bitcoin', 'tether')

selected_currency=st.selectbox('Select dataset for prediction', crypto_currencies)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# we need a function load_data()
@st.cache

def load_data(n_rows):
    data = pd.read_csv('data_advanced.csv') #(n_rows =n_rows))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data...')
data = load_data(selected_currency)
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data.tail())

def my_widget(key):
    st.subheader('Hello there!')
    return st.button("Click me " + key)

# This works in the main area
clicked = my_widget("first")

# And within an expander
my_expander = st.expander("Expand", expanded=True)
with my_expander:
    clicked = my_widget("second")

# AND in st.sidebar!
with st.sidebar:
    clicked = my_widget("third")

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['price_usd']))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


#Forecasting

df_train = data[['datetime']]
df_train = df_train.rename(columns={'datetime': 'ds'})

#testing out the metric feature
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")
st.metric(label="Gas price", value=4, delta=-0.5,
     delta_color="inverse")

st.metric(label="Active developers", value=123, delta=123,
     delta_color="off")

# adding a chart

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# Mutating chart


# Now the table shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

#m = Prophet()
#m.fit(df_train)
#future = m.make_future_dataframe(periods=period)
#forecast = m.predict(future)

#st.subheader('Forecast data')
#st.write(forecast.tail())


#st.write('forecast data')
#fig1 = plot_plotly(m, forecast)
#st.plotly_chart(fig1)

#st.write('forecast components')
#fig2 = m.plot_components(forecast)
#st.write(fig2)
