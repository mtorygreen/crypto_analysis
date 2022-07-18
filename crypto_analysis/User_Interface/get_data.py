import pandas as pd
import yfinance as yf
import san
from datetime import  datetime


def load_data(ticker=None, timeframe='5y'):
    '''
    This function loads stcks data per API call.
    '''
    print(ticker)
    data = yf.download(ticker, period=timeframe)
    data.reset_index(inplace=True)
    print(data)
    df_stocks.set_index('Date', inplace=True)
    df_stocks = df_stocks['Open']
    df_stocks = df_stocks.reset_index(level=0)
    df_stocks = df_stocks.rename(columns={'Date': 'datetime'})


    return df_stocks





if __name__ == "__main()__":
    print("=====starting=====")
    # variables
    timeframe = '5y'
    stocks = ('^GSPC', '^IXIC', '000001.SS', '^N100', '^NDX')

    df_stocks = load_data(ticker=stocks)
    print(df_stocks)
