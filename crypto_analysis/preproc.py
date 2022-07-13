import os
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage

# Extract the current directory
os.getcwd()

# os.path.join is used to replace the need of using special characters,
# such as `/` between folders The point is to make it universal,
# since different OS use different characters.
# We move back once with `..` and enter into the `raw_data` folder.

csv_path = os.path.join('..','raw_data')

# filenames of the selected features and target .csv files

files = ['Price.csv', 'Volume.csv', 'Daily Active Addresses.csv',
         'Network Growth.csv', 'Transaction Volume USD.csv',
         'Velocity.csv', 'Average Fees USD.csv',
         'Dev Activity Contributors Count.csv',
         'Development Activity.csv',
         'Social Dominance.csv', 'Social Volume.csv',
         'Twitter Followers.csv', 'Weighted Sentiment.csv',
         'Whales Transactions 100K+.csv',
         'Whale Transactions 1M +.csv'
         ]

# LocalPath
LOCAL_PATH='raw_data/data.csv'
# project id
PROJECT_ID='lewagon-913-crypto-analysis'
# bucket name
BUCKET_NAME='crypto913'
# bucket directory in which to store the uploaded file
# (we choose to name this data as a convention)
BUCKET_FOLDER='data'
REGION='europe-west1'

def data_fromcsv():
    '''Function that read the selected csv files
    and save the merge of them into raw_data/data.csv
    and return data dataset'''

    dataset_1 = pd.read_csv(os.path.join('..','raw_data', 'Price.csv'))
    dataset_2 = pd.read_csv(os.path.join('..','raw_data', 'Volume.csv'))
    dataset_3 = pd.read_csv(os.path.join('..','raw_data', 'Daily Active Addresses.csv'))
    dataset_4 = pd.read_csv(os.path.join('..','raw_data', 'Network Growth.csv'))
    dataset_5 = pd.read_csv(os.path.join('..','raw_data', 'Transaction Volume USD.csv'))
    dataset_6 = pd.read_csv(os.path.join('..','raw_data', 'Velocity.csv'))
    dataset_7 = pd.read_csv(os.path.join('..','raw_data', 'Average Fees USD.csv'))
    dataset_8 = pd.read_csv(os.path.join('..','raw_data', 'Dev Activity Contributors Count.csv'))
    dataset_9 = pd.read_csv(os.path.join('..','raw_data', 'Development Activity.csv'))
    dataset_10 = pd.read_csv(os.path.join('..','raw_data', 'Social Dominance.csv'))
    dataset_11 = pd.read_csv(os.path.join('..','raw_data', 'Social Volume.csv'))
    dataset_12 = pd.read_csv(os.path.join('..','raw_data', 'Twitter Followers.csv'))
    dataset_13 = pd.read_csv(os.path.join('..','raw_data', 'Weighted Sentiment.csv'))
    dataset_14 = pd.read_csv(os.path.join('..','raw_data', 'Whales Transactions 100K+.csv'))
    dataset_15 = pd.read_csv(os.path.join('..','raw_data', 'Whale Transactions 1M +.csv'))

    data = pd.merge(dataset_1, dataset_2).merge(dataset_3).merge(
                dataset_4).merge(dataset_5).merge(
                    dataset_6).merge(dataset_7).merge(
                        dataset_8).merge(dataset_9).merge(
                            dataset_10).merge(dataset_11).merge(
                                dataset_12).merge(
                                    dataset_13).merge(
                                        dataset_14).merge(dataset_15)
    data.to_csv(os.path.join('..','raw_data', 'data.csv'))
    return data


# def export_datacsv():
#     '''
#     Function that create data.csv file into raw_data folder
#     '''
#     data = data_fromcsv()
#     data.to_csv(os.path.join('..','raw_data', 'data.csv'))

def storage_upload(data=BUCKET_FOLDER, bucket=BUCKET_NAME):
    '''
    Perform the upload of data.csv into the bucket folder of the gcp project
    '''
    client = storage.Client().bucket(bucket)
    storage_location = "{}/{}".format(data, "data.csv")
    blob = client.blob(storage_location)
    blob.upload_from_filename(os.path.join('..','raw_data', 'data.csv'))


if __name__ == '__main__':
    data = data_fromcsv()
    storage_upload()
    print(data.head())
    print(data.isnull().sum().sort_values(ascending=False))
