import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MAPE
from tensorflow.keras import callbacks
from google.cloud import storage
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[WW] [DubaiMunichTokyoAix] [CryptoTeamGHHG] cryptoanalysis + 0.1"

class Trainer():
    def __init__(self, data):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.data = data
        self.experiment_name = EXPERIMENT_NAME

    def subsample_sequence(df, length):
        '''
        function that return a random slice of features and targets
        len(X) = lenght and len(y) = 3
        '''
        last_possible = df.shape[0] - length - 3
        random_start = np.random.randint(0, last_possible)
        X = df[random_start: random_start+length].values
        y = df.iloc[random_start+length:random_start+length+3][['price_usd']]
        return X, y

    def get_X_y(df, length_of_observations):
        '''
        function that returns a list of random slices of features and targets
        len(X[0]) = lenght and len(y[0]) = 3
        '''
        X, y = [], []
        for length in length_of_observations:
            xi, yi = subsample_sequence(df, length)
            X.append(xi)
            y.append(yi)
        return X, y

    def split_tr_te(df, horizon=3, ratio=0.95):
        '''
        function that returns a training and test set
        arguments are:
        the horizon of prediction
        the ratio of the train/test split
        '''
        # the gap to avoid data leakage
        # gap = horizon - 1
        # len_ = int(ratio*df.shape[0])
        # data_train = df[:len_]
        # data_test = df[len_+gap:]
        # return data_train, data_test

        len_ = int(ratio*df.shape[0])
        data_train = df[:len_]
        data_test = df[len_+1:]
        return data_train, data_test

        # len_ = 1696
        # data_train = df[:len_]
        # data_test = df[len_:]
        # return data_train, data_test

    def extract_xy_tr_te(train,
                        test,
                        train_splits = 100,
                        train_time_min = 79,
                        train_time_max = 81):
        '''
        function returns a serie of train and test data
        train splits is the number of selections of our dataset
        train_time_min is the minimum number of days that are randomly choosen by the get_X_y function
        train_time_max is the maximum number of days that are randomly choosen by the get_X_y function
        '''
        length_of_observations = np.random.randint(train_time_min, train_time_max, train_splits)
        X_train, y_train = get_X_y(train, length_of_observations)
        #length_of_observations = np.random.randint(train_time_min, train_time_max, train_splits)
        #X_test, y_test = get_X_y(test, length_of_observations)
        return X_train, y_train #, X_test, y_test

    def padding_seq(train):
        '''
        function that return the padded version of the train dataset
        to uniform the size of the model imput
        '''
        return pad_sequences(train, dtype='float32', value=-1)

    def baseline_model(X_train_pad, y_train):
        '''
        function that return a trained baseline model and its fitting history
        and save locally the trained model file basemodel.joblib
        '''
        normalizer = Normalization()
        normalizer.adapt(X_train_pad)
        model = Sequential()
        model.add(normalizer)
        model.add(layers.Masking(mask_value=-1))
        model.add(layers.GRU(40, activation='tanh')) # GS N UNITS
        model.add(layers.Dense(40, activation='relu')) # GS N UNITS
        model.add(layers.Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.01), metrics=MAPE)
        es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        history = model.fit(X_train_pad,
                    np.array(y_train),
                    epochs=200,
                    batch_size=128,
                    validation_split=0.3,
                    callbacks=[es],
                    verbose=1)
        joblib.dump(model, 'basemodel.joblib')
        return model, history

    def plot_history(history, title='', axs=None, exp_name=""):
        '''
        return the loss and metric plots of train and test fit process
        '''
        if axs is not None:
            ax1, ax2 = axs
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if len(exp_name) > 0 and exp_name[0] != '_':
            exp_name = '_' + exp_name
        ax1.plot(history.history['loss'], label = 'train' + exp_name)
        ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
        ax1.set_ylim(0., 100000)
        ax1.set_title('loss')
        ax1.legend()

        ax2.plot(history.history['mean_absolute_percentage_error'], label='train mape'  + exp_name)
        ax2.plot(history.history['val_mean_absolute_percentage_error'], label='val mape'  + exp_name)
        ax1.set_ylim(0., 100000)
        ax2.set_title('mape')
        ax2.legend()
        return (ax1, ax2)

    def pred_3d_price(model, test):
        '''
        return the prediction of three days after the test data
        '''
        return model.predict(test)

    # PARAMETERS FOR GCP BASEMODEL UPLOAD

    STORAGE_LOCATION = 'models/basemodel.joblib'
    BUCKET_NAME='crypto913'

    def upload_model_to_gcp():
        '''
        function that upload the trained model to gcp
        '''
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('basemodel.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    trainer = Trainer(X=X_train, y=y_train)
    # evaluate
    trainer.evaluate(X_test=X_val, y_test=y_val)

    # save model
    trainer.save_model()
