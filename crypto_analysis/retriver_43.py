import pandas as pd
import yfinance as yf
import san
from datetime import  datetime
import requests
import json
import os

pd.options.mode.chained_assignment = None  # default='warn'

class Retriver():
    def __init__(self):
        """
        Retriver class to extract data from stocks API
        """
        # Class variables for yfinance
        self.timeframe = '6y'
        self.stocks = ('^GSPC', '^IXIC', '000001.SS', '^N100', '^NDX')
        # Class variables for sanbase
        self.free_metrics = ['price_usd',
                             'volume_usd',
                             'twitter_followers',
                             'daily_opening_price_usd',
                             'daily_high_price_usd',
                             'daily_low_price_usd',
                             'daily_closing_price_usd',
                             'daily_avg_price_usd',
                             'daily_trading_volume_usd',
                             'marketcap_usd',
                             'daily_active_addresses',
                             'dev_activity',
                             'dev_activity_contributors_count',
                             'social_volume_total'
                             ]
        self.all_metrics = ['price_usd',
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
        self.year = 6
        self.days = 365
        self.tketh = 'ethereum'
        self.tkbtc = 'bitcoin'
        self.interval = '1d'
        # Class variables for defillama
        self.url_total_TVL = 'https://api.llama.fi/charts'
        self.url_eth_TVL = 'https://api.llama.fi/charts/Ethereum'
        # Class variables for cryptoslam
        self.url_cs_glb = 'https://api2.cryptoslam.io/api/nft-indexes/nftglobal'
        self.url_cs_eth = 'https://api2.cryptoslam.io/api/nft-indexes/Ethereum'
        # Class variables for etherscan
        self.url_es = 'https://etherscan.io/chart/tx?output=csv'
        # Class variable for request settings
        self.setup_request = f'Mozilla/5.0 (X11; Ubuntu;' \
                             f'Linux x86_64; rv:77.0) ' \
                             f'Gecko/20100101 Firefox/77.0'

    def load_data_stocks(self):
        '''
        This function loads stocks data from yfinance.
        '''
        self.df_yf = yf.download(self.stocks, period=self.timeframe)
        self.df_yf.reset_index(inplace=True)
        self.df_yf.set_index('Date', inplace=True)
        self.df_yf = self.df_yf['Open']
        self.df_yf = self.df_yf.reset_index(level=0)
        self.df_yf = self.df_yf.rename(columns={'Date': 'datetime'})
        print('yfinance:    100%')

    def load_data_sanbase(self):
        '''
        This function loads sentiment data from sanbase.
        '''
        san.ApiConfig.api_key = ''
        # Initialize empty DataFrames to collect data
        self.df_eth = pd.DataFrame()
        self.df_btc = pd.DataFrame()
        # Extract df_eth dataset
        for metric in self.all_metrics:
            self.df_eth[metric] = san.get(
                metric + '/' + self.tketh,
                from_date=f"utc_now - {self.year*self.days}d",
                to_date="utc_now",
                interval=self.interval)
        # Interpolating twitter followers before data starts
        self.df_eth['twitter_followers'].iloc[0] = 0
        self.df_eth['twitter_followers'] = self.df_eth[
            'twitter_followers'].interpolate()
        # Extract df_btc dataset
        self.df_btc['price_usd'] = san.get(
            'price_usd' + '/' + self.tkbtc,
            from_date=f"utc_now - {self.year*self.days}d",
            to_date="utc_now",
            interval=self.interval)
        # Customizing column name for BTC usd price
        self.df_btc = self.df_btc.rename(columns={
            'price_usd': 'btc_price_usd'})
        # Combine cryprocurrencies into one DataFrame
        self.df_crypto = pd.merge(self.df_eth, self.df_btc,
                                  how='outer', on='datetime')
        self.df_crypto.index = pd.to_datetime(self.df_crypto.index)
        self.df_crypto = self.df_crypto.reset_index(level=0)
        self.df_crypto['datetime'] = self.df_crypto['datetime'].apply(
            lambda x: x.replace(tzinfo=None))
        print('sanbase:     100%')

    def load_data_defillama(self):
        '''
        Loading total_TVL and eth_TVL data from http requests
        '''
        r_total_TVL = requests.get(self.url_total_TVL)
        r_eth_TVL = requests.get(self.url_eth_TVL)
        json_total_TVL = r_total_TVL.json()
        json_eth_TVL = r_eth_TVL.json()
        # Creating DataFrames
        self.df_total_TVL = pd.DataFrame(json_total_TVL)
        self.df_eth_TVL = pd.DataFrame(json_eth_TVL)
        # Renaming columns
        self.df_total_TVL = self.df_total_TVL.rename(columns={
            'totalLiquidityUSD': 'total_TVL', 'date': 'datetime'})
        self.df_eth_TVL = self.df_eth_TVL.rename(columns={
            'totalLiquidityUSD': 'eth_TVL', 'date': 'datetime'})
        # Merging total_TVL & eth_TVL
        self.df_tvl = pd.merge(self.df_total_TVL, self.df_eth_TVL,
                               how='outer', on='datetime')
        self.df_tvl['datetime'] = self.df_tvl['datetime'].apply(
            lambda x:datetime.utcfromtimestamp(int(x)).strftime(
                '%Y-%m-%d %H:%M:%S'))
        self.df_tvl['datetime'] = pd.to_datetime(self.df_tvl['datetime'])
        print('defillama:   100%')

    def load_data_cryptoslam(self):
        '''
        Loading data from Cryptoslam http requests
        '''
        # Making function works in Python 2 and 3
        try:
            from urllib.request import Request, urlopen  # Python 3
        except ImportError:
            from urllib2 import Request, urlopen  # Python 2
        csrglb = Request(self.url_cs_glb)
        csreth = Request(self.url_cs_eth)
        csrglb.add_header('User-Agent', self.setup_request)
        uo_glb = urlopen(csrglb)
        csreth.add_header('User-Agent', self.setup_request)
        uo_eth = urlopen(csreth)
        dataglb = json.load(uo_glb)
        dataeth = json.load(uo_eth)
        # Extract useful global NFT data
        glbdays  = []
        glbprice = []
        glbub    = []
        for glbk, glbv in dataglb.items():
            for glbk1, glbv1 in glbv.items():
                if glbk1 == 'dailySummaries':
                    for glbk2, glbv2 in glbv1.items():
                        glbdays.append(glbk2)
                        glbprice.append(glbv2['totalPriceUSD'])
                        glbub.append(glbv2['uniqueBuyers'])
        df_cs_glb = pd.DataFrame()
        df_cs_glb['datetime'] = glbdays
        df_cs_glb['total_NFT_sales'] = glbprice
        df_cs_glb['total_NFT_buyers'] = glbub
        # Extract useful ETH NFT data
        ethdays  = []
        ethprice = []
        ethub    = []
        for ethk, ethv in dataeth.items():
            for ethk1, ethv1 in ethv.items():
                if ethk1 == 'dailySummaries':
                    for ethk2, ethv2 in ethv1.items():
                        ethdays.append(ethk2)
                        ethprice.append(ethv2['totalPriceUSD'])
                        ethub.append(ethv2['uniqueBuyers'])
        df_cs_eth = pd.DataFrame()
        df_cs_eth['datetime'] = ethdays
        df_cs_eth['eth_NFT_sales'] = ethprice
        df_cs_eth['eth_NFT_buyers'] = ethub
        # Merging datasets
        self.df_NFT = pd.merge(df_cs_glb, df_cs_eth, how='outer', on='datetime')
        self.df_NFT['datetime'] = pd.to_datetime(self.df_NFT['datetime'])
        print('cryptoslam:  100%')

    def load_data_etherscan(self):
        '''
        Loading Daily transaction via csv request
        '''
        # Making function works in Python 2 and 3
        try:
            from urllib.request import Request, urlopen  # Python 3
        except ImportError:
            from urllib2 import Request, urlopen  # Python 2
        req = Request(self.url_es)
        req.add_header('User-Agent', self.setup_request)
        content = urlopen(req)
        self.daily_transactions = pd.read_csv(content)
        self.daily_transactions = self.daily_transactions.rename(columns={
            'Date(UTC)': 'datetime', 'UnixTimeStamp':
            'timestamp', 'Value': 'daily transactions'})
        self.daily_transactions.drop(columns='timestamp', inplace=True)
        self.daily_transactions['datetime']=self.daily_transactions[
            'datetime'].astype('datetime64')
        print('etherscan:   100%')

    def data_merge(self):
        '''
        merging all the dataframes into one Dataframe
        '''
        #WITH df_NFT UNCOMMENT THE LINES AFTER
        #self.data = pd.merge(self.df_NFT, self.df_crypto, how='left',
        #                     on='datetime')
        #self.data = pd.merge(self.data, self.df_tvl, how='left', on='datetime')
        #WITH df_NFT COMMENT THE 2 NEXT LINES
        self.data = pd.merge(self.df_crypto, self.df_tvl, how='left',
                             on='datetime')
        self.data = pd.merge(self.data, self.df_NFT, how='left', on='datetime')
        self.data = pd.merge(self.data, self.df_yf, how='left', on='datetime')
        self.data = pd.merge(self.data, self.daily_transactions,
                             how='left', on='datetime')

    def data_engine(self):
        '''
        Function to interpolate stock weekends
        and perform data engineering on some features
        '''
        # Interpolation
        features_to_interpolate = (['000001.SS',
                                    '^GSPC',
                                    '^IXIC',
                                    '^N100',
                                    '^NDX',
                                    'twitter_followers',
                                    'eth_NFT_sales',
                                    'eth_NFT_buyers',
                                    'total_NFT_sales',
                                    'total_NFT_buyers'])
        for feature in features_to_interpolate:
            self.data[feature] = self.data[feature].interpolate()
        # Engineering
        self.data["volume_ETH"] = self.data["volume_usd"]/self.data["price_usd"]
        self.data["daily_trading_volume_ETH"] = self.data[
            "daily_trading_volume_usd"]/self.data["price_usd"]
        self.data["eth_NFT_sales_ETH"] = self.data[
            "eth_NFT_sales"]/self.data["price_usd"]
        self.data["total_NFT_sales_ETH"] = self.data[
            "total_NFT_sales"]/self.data["price_usd"]
        self.data["total_TVL_ETH"] = self.data[
            "total_TVL"]/self.data["price_usd"]
        self.data["eth_TVL_ETH"] = self.data["eth_TVL"]/self.data["price_usd"]
        if "fees_usd" in self.data.columns:
            self.data["fees_usd_ETH"] = self.data["fees_usd"]/self.data[
                "price_usd"]
        if "mvrv_usd" in self.data.columns:
            self.data["mvrv_usd_ETH"] = self.data["mvrv_usd"]/self.data[
                "price_usd"]
        # Drop columns
        self.data = self.data.drop(columns=['fees_usd',
                                            'mvrv_usd',
                                            'volume_usd',
                                            'daily_trading_volume_usd',
                                            'eth_NFT_sales',
                                            'total_NFT_sales',
                                            'total_TVL',
                                            'eth_TVL'], errors='ignore')
        # Interpolation after Engineering
        features_to_interpolate = (['social_volume_total',
                                    'total_TVL_ETH',
                                    'eth_TVL_ETH'])
        for feature in features_to_interpolate:
            self.data[feature].iloc[0] = 0
            self.data[feature] = self.data[feature].interpolate()
        # Take into account only the last 5 years of data
        # and drop the incomplete day
        self.data = self.data.tail(365*5+1).head(365*5)
        self.data.to_csv(os.path.join('raw_data', 'data_43.csv'))

if __name__ == "__main__":
    retriver = Retriver()
    retriver.load_data_stocks()
    retriver.load_data_sanbase()
    retriver.load_data_defillama()
    retriver.load_data_cryptoslam()
    retriver.load_data_etherscan()
    retriver.data_merge()
    retriver.data_engine()
