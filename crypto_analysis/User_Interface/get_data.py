from dataclasses import dataclass
import pandas as pd
import yfinance as yf
import san
from datetime import  datetime
import requests


def load_data_stocks(ticker=None, timeframe='5y'):
    '''
    This function loads stocks data per API call.
    '''

    data = yf.download(ticker, period=timeframe)
    data.reset_index(inplace=True)

    df_stocks = data
    df_stocks.set_index('Date', inplace=True)
    df_stocks = df_stocks['Open']
    df_stocks = df_stocks.reset_index(level=0)
    df_stocks = df_stocks.rename(columns={'Date': 'datetime'})


    return df_stocks

def load_data_sanbase(year=10, days=365, tketh = 'ethereum', tkbtc = 'bitcoin'):
    '''
    This function loads sanbase data per API call.
    '''

    san.ApiConfig.api_key = '3e4ne7awzdf2yy65_v7liam22jvyryahn'

    future_metrics = ['whale_transaction_count_100k_usd_to_inf_change_1d',
                    'whale_transaction_count_1m_usd_to_inf_change_30d']

    old_metrics = ['price_usd', 'volume_usd',
                    'daily_active_addresses', 'network_growth',
                    'transaction_volume_usd', 'transaction_volume',
                    'velocity', 'dev_activity', 'social_dominance_total',
                    'social_volume_total', 'twitter_followers',
                    'sentiment_positive_total', 'sentiment_negative_total',
                    ]

    metrics = ['price_usd',
            'volume_usd',
            'twitter_followers',
            'daily_opening_price_usd',
            'daily_high_price_usd',
            'daily_low_price_usd',
            'daily_closing_price_usd',
            'daily_avg_price_usd',
            'daily_trading_volume_usd',
            'marketcap_usd',
            'mvrv_usd',
            'nvt',
            'circulation_1d',
            'dormant_circulation_90d',
            'exchange_balance',
            'daily_active_addresses',
            'network_growth',
            'transaction_volume',
            'fees_usd',
            'velocity',
            'dev_activity',
            'dev_activity_contributors_count',
            'sentiment_positive_total',
            'sentiment_negative_total',
            'sentiment_balance_total',
            'social_dominance_total',
            'social_volume_total',
            'unique_social_volume_total_5m',
            'whale_transaction_count_100k_usd_to_inf',
            'whale_transaction_count_1m_usd_to_inf'
            ]


    # Initialize empty DataFrames to collect data
    df_eth = pd.DataFrame()
    df_btc = pd.DataFrame()

    for metric in metrics:
        df_eth[metric] = san.get(
            metric + '/' + tketh,
            from_date=f"utc_now - {year*days}d",
            to_date="utc_now",
            interval="1d"
        )
    df_eth['twitter_followers'].iloc[0] = 0
    df_eth['twitter_followers'] = df_eth['twitter_followers'].interpolate()

    df_btc['price_usd'] = san.get(
        'price_usd' + '/' + tkbtc,
        from_date=f"utc_now - {year*days}d",
        to_date="utc_now",
        interval="1d"
    )

    df_btc = df_btc.rename(columns={'price_usd': 'btc_price_usd'})

    # Combine cryprocurrencies into one DataFrame
    df_crypto = pd.merge(df_eth, df_btc, how='outer', on='datetime')
    df_crypto.index = pd.to_datetime(df_crypto.index)
    df_crypto = df_crypto.reset_index(level=0)
    df_crypto['datetime'] = df_crypto['datetime'].apply(lambda x: x.replace(tzinfo=None))


    return df_crypto

def load_defillama_data(url_total_TVL = 'https://api.llama.fi/charts', url_eth_TVL = 'https://api.llama.fi/charts/Ethereum'):
    '''
    Loading data through API call total_TVL and eth_TVL
    '''

    r_total_TVL = requests.get(url_total_TVL)
    r_eth_TVL = requests.get(url_eth_TVL)

    json_total_TVL = r_total_TVL.json()
    json_eth_TVL = r_eth_TVL.json()

    # create DataFrame
    df_total_TVL = pd.DataFrame(json_total_TVL)
    df_eth_TVL = pd.DataFrame(json_eth_TVL)

    # rename columns
    df_total_TVL = df_total_TVL.rename(columns={'totalLiquidityUSD': 'total_TVL', 'date': 'datetime'})
    df_eth_TVL = df_eth_TVL.rename(columns={'totalLiquidityUSD': 'eth_TVL', 'date': 'datetime'})

    # Merging total_TVL & eth_TVL
    df_tvl = pd.merge(df_total_TVL, df_eth_TVL, how='outer', on='datetime')

    df_tvl['datetime'] = df_tvl['datetime'].apply(lambda x:datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    df_tvl['datetime'] = pd.to_datetime(df_tvl['datetime'])

    return df_tvl

# Path finden (os.path.join)
def load_cryptoslam_data(eth_NFT_sales = pd.read_csv('ETH NFT Sales.csv'),
                        total_NFT_sales = pd.read_csv('Total NFT Sales.csv')):

    '''
    Loading data from Cryptoslam from a CSV files
    '''

    eth_NFT_sales = eth_NFT_sales.rename(columns={'Sales (USD) (y)': 'eth_NFT_sales', 'Unique Buyers (y)': 'eth_NFT_buyers', 'DateTime': 'datetime'})
    total_NFT_sales = total_NFT_sales.rename(columns={'Sales (USD) (y)': 'total_NFT_sales', 'Unique Buyers (y)': 'total_NFT_buyers', 'DateTime': 'datetime'})

    df_NFT = pd.merge(eth_NFT_sales, total_NFT_sales, how='outer', on='datetime')
    df_NFT['datetime'] = pd.to_datetime(df_NFT['datetime'])

    return df_NFT

def load_etherscan_data():
    '''
    Loading Daily transaction via csv
    '''
    daily_transactions = pd.read_csv('Daily Transactions_7.15.22.csv')
    daily_transactions = daily_transactions.rename(columns={'Date(UTC)': 'datetime', 'UnixTimeStamp': 'timestamp', 'Value': 'daily transactions'})
    daily_transactions.drop(columns='timestamp', inplace=True)
    daily_transactions['datetime']=daily_transactions['datetime'].astype('datetime64')

    return daily_transactions

def data_merged(df_crypto=None, df_NFT=None, df_tvl=None, df_stocks=None, daily_transactions=None):

    '''
    merging all the dataframes into one Dataframe
    '''
    # data = pd.DataFrame()


    data = pd.merge(df_crypto, df_NFT, how='left', on='datetime')
    data = pd.merge(data, df_tvl, how='left', on='datetime')
    data = pd.merge(data, df_stocks, how='left', on='datetime')
    data = pd.merge(data, daily_transactions, how='left', on='datetime')


    return data


def adapting_data(data=None):

    '''
    converting_USD_in_ETH
    '''
    data["volume_ETH"] = data["volume_usd"]/data["price_usd"]
    data["daily_trading_volume_ETH"] = data["daily_trading_volume_usd"]/data["price_usd"]
    data["eth_NFT_sales_ETH"] = data["eth_NFT_sales"]/data["price_usd"]
    data["total_NFT_sales_ETH"] = data["total_NFT_sales"]/data["price_usd"]
    data["total_TVL_ETH"] = data["total_TVL"]/data["price_usd"]
    data["eth_TVL_ETH"] = data["eth_TVL"]/data["price_usd"]
    data["fees_usd_ETH"] = data["fees_usd"]/data["price_usd"]
    data["mvrv_usd_ETH"] = data["mvrv_usd"]/data["price_usd"]

    # dropping columns
    df = data.drop(columns=['fees_usd', 'mvrv_usd', 'volume_usd', 'daily_trading_volume_usd','eth_NFT_sales','total_NFT_sales','total_TVL','eth_TVL'])

    return df


def load_data():
    timeframe = '5y'
    stocks = ('^GSPC', '^IXIC', '000001.SS', '^N100', '^NDX')
    df_stocks = load_data_stocks(ticker=stocks)

    df_crypto = load_data_sanbase(year=5, days=365)

    df_tvl = load_defillama_data()

    df_NFT = load_cryptoslam_data()

    daily_transactions = load_etherscan_data()

    merged_data = data_merged(df_crypto=df_crypto, df_NFT=df_NFT, df_tvl=df_tvl, df_stocks=df_stocks, daily_transactions=daily_transactions)

    # Final DataFrame with all the data from different sources
    df = adapting_data(data=merged_data)
    return df




if __name__ == "__main__":

    timeframe = '5y'
    stocks = ('^GSPC', '^IXIC', '000001.SS', '^N100', '^NDX')
    df_stocks = load_data_stocks(ticker=stocks)


    df_crypto = load_data_sanbase(year=5, days=365)


    df_tvl = load_defillama_data()


    df_NFT = load_cryptoslam_data()


    daily_transactions = load_etherscan_data()


    merged_data = data_merged(df_crypto=df_crypto, df_NFT=df_NFT, df_tvl=df_tvl, df_stocks=df_stocks, daily_transactions=daily_transactions)

    # Final DataFrame with all the data from different sources
    df = adapting_data(data=merged_data)
