import streamlit as st
import requests
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import san
from get_data import load_data


#testing out
import altair as alt





# import final_python_file
# from final_python_file import load_data
# df_all_data = load_data



# Title st.title('Cryptocurrencies Prediction App')

st.title('Cryptocurrencies Prediction App')

# time
START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

# Data for Graph


crypto_currencies=('ethereum', 'bitcoin', 'tether')

#selected_currency=st.selectbox('Select dataset for prediction', crypto_currencies)

# All the data from different sources in one DataFrame
df = load_data()

#my_expander = st.expander(#)
#my_expander.write(df.head(n=10))
#clicked = my_expander.button('Show raw data')




# Plot Graph
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['price_usd']))
    fig.layout.update(title_text='Time Series Ethereum Price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()







# Under the Graph show Metric of our prediction for the next three days compared to the value t -1

#st.metric(label="Ethereum", value="$1.109,67", delta="+ 0.01")
col1, col2, col3 = st.columns(3)
col1.metric(label="In One Day", value="$1.109,67", delta="+ 0.01")
col2.metric(label="In Two Days", value="$1.109,67", delta="+ 0.01")
col3.metric(label="In Three Days", value="$1.109,67", delta="- 0.01", delta_color="normal")




st.subheader('Raw data')
st.write(df.tail())

if __name__ == "__main__":
    data = load_data()
    print(data.shape)
