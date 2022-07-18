import streamlit as st
from datetime import date
import yfinance as yf
#from fbprophet import Prophet
import pandas as pd
#from prophet import Prophet
#from prophet.plot import plot_plotly
from plotly import graph_objs as go
import altair as alt
from vega_datasets import data



#START = '2015-01-01'
#TODAY = date.today().strftime('%Y-%m-%d')

#st.title('Stock Prediction App')
#stocks=('AAPL', 'GOOG', 'MSFT', 'GME')

#selected_stocks=st.selectbox('Select dataset for prediction', stocks)

#n_years = st.slider('Years of prediction:', 1, 4)
#period = n_years * 365

# we need a function load_data()
@st.cache

#def load_data(ticker):
    #data = yf.download(ticker, START, TODAY) #(n_rows =n_rows))
    #data.reset_index(inplace=True)
    #return data

#data_load_state = st.text('Load data...')
#data = load_data(selected_stocks)
#data_load_state.text('Loading data...done!')

#st.subheader('Raw data')
#st.write(data.tail())

#def plot_raw_data():
 #   fig = go.Figure()
  #  fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
   # fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    #fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
 #   st.plotly_chart(fig)

#plot_raw_data()

#Forecasting

#df_train = data[['Date', 'Close']]
#df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

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


#'''trying out Time series annotations'''


@st.experimental_memo
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source

source = get_data()

# Original time series chart. Omitted `get_chart` for clarity
chart = get_chart(source)

# Input annotations
ANNOTATIONS = [
    ("Mar 01, 2008", "Pretty good day for GOOG"),
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
    ("Nov 01, 2008", "Market starts again thanks to..."),
    ("Dec 01, 2009", "Small crash for GOOG after..."),
]

# Create a chart with annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
annotations_df.date = pd.to_datetime(annotations_df.date)
annotations_df["y"] = 0
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=15, text="", dx=5, dy=9, align="center")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)

# Display both charts together
st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)
