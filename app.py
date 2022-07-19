import streamlit as st
import requests
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import san
#from get_data import load_data
from crypto_analysis import trainer_advanced


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
#df = load_data()
data = pd.read_csv('raw_data/data_advanced_v2.csv')
trainer = trainer_advanced.Trainer(data)
trainer.preproc_data()

#my_expander = st.expander(#)
#my_expander.write(df.head(n=10))
#clicked = my_expander.button('Show raw data')

# Plot Graph
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,
                             y=data['price_usd']))
    fig.layout.update(title_text='Time Series Ethereum Price',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Compute the prediction and extraxt next three days in pred list
trainer.extract_xy_tr_te()
trainer.padding_seq()
pred = trainer.get_prediction()

last_known_days = trainer.target_scaler.inverse_transform(
    np.array(
        trainer.X_test_pad)[0,:,0].reshape(-1,1))

# Under the Graph show Metric of our prediction for the next three days compared to the value t -1

#st.metric(label="Ethereum", value="$1.109,67", delta="+ 0.01")
col1, col2, col3 = st.columns(3)

col1.metric(label="In One Day", value=f'${pred[0,0].item()}',
            delta=pred[0,0].item() - last_known_days[4,0].item())

col2.metric(label="In Two Days", value=f'${pred[1,0].item()}',
            delta=pred[1,0].item() - pred[0,0].item())

col3.metric(label="In Three Days", value=f'${pred[2,0].item()}',
            delta=pred[2,0].item() - pred[1,0].item(), delta_color="normal")

st.subheader('Raw data')
st.write(data.tail())

#if __name__ == "__main__":
    #data = load_data()
    #print(data.shape)
