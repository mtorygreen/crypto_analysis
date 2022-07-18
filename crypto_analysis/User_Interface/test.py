import streamlit as st
import requests
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import san

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

@st.experimental_memo
@st.cache

def load_data():
    data = pd.read_csv('data_advanced.csv') #(n_rows =n_rows))
    data.reset_index(inplace=True)
    return data
# Original time series chart. Omitted `get_chart` for clarity



#data_load_state = st.text('Load data...')
data = load_data()
#data_load_state.text('Loading data...done!')

#chart = get_chart(data)


# Plot Graph
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['price_usd']))
    fig.layout.update(title_text='Time Series Ethereum Price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

chart = plot_raw_data()


# Input annotations
ANNOTATIONS = [
    ("2017-07-21", "Pretty good day for GOOG"),
    ("2022-07-14", "Something's going wrong for GOOG & AAPL"),
]

# Create a chart with annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
annotations_df.date = pd.to_datetime(annotations_df.date)
annotations_df["y"] = 0
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=15, text="⬇", dx=0, dy=-10, align="center")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)

# Display both charts together
st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)



# Under the Graph show Metric of our prediction for the next three days compared to the value t -1

#st.metric(label="Ethereum", value="$1.109,67", delta="+ 0.01")
col1, col2, col3 = st.columns(3)
col1.metric(label="In One Day", value="$1.109,67", delta="+ 0.01")
col2.metric(label="In Two Days", value="$1.109,67", delta="+ 0.01")
col3.metric(label="In Three Days", value="$1.109,67", delta="- 0.01", delta_color="normal")




st.subheader('Raw data')
st.write(data.tail())

if __name__ == "__main__":
    data = load_data()
    print(data.shape)
