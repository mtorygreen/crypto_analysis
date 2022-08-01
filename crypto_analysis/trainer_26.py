import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import MAPE, mse, RootMeanSquaredError
from tensorflow.keras import callbacks
from tensorflow.keras.backend import clear_session
from google.cloud import storage
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, TimeDistributed, RepeatVector
import datetime

pd.options.mode.chained_assignment = None  # default='warn'

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[WW] [DubaiMunichTokyoAix] [CryptoTeamGHHG] cryptoanalysis + 0.1"
e = datetime.datetime.now()
date = f'{str(e.year)[-2:]}_{e.month}_{e.day}_{e.hour}_{e.minute}_{e.second}'

STORAGE_LOCATION = f'models/model_{date}.joblib'
BUCKET_NAME='crypto913'


class Trainer():
    def __init__(self, data):
        """
        trainer class needed for mlflow
        need a data DataFrame from the advanced_data.py
        """
        self.model = Sequential()
        self.data = data
        # MlFlow EXPERIMET NAME
        self.experiment_name = EXPERIMENT_NAME

        # days of X_test
        self.memory = 3

        # functions parameters:
        self.train_splits = 20000
        self.train_time_max = self.memory

        self.rnn_layer1 = 300
        self.dense_layer1 = 100
        self.dense_layer2 = 20
        self.dense_output = 1                #OK 1
        self.compile_optimizer = 'adam'      #OK adam
        self.compile_loss = 'mse'            #OK mse
        self.compile_learning_rate = 0.0001  #0.00001 OK but slow improvement for each bach
        self.es_patience = 400               #OK 100
        self.fit_epochs = 100                #OK 3000 - 5000
        self.fit_bach_size = 8               #OK 16 32
        self.fit_validation_split = 0.3      #OK 0.3

    def preproc_data(self):
        # Data Cleaning
        self.data.set_index('datetime', inplace=True)
        self.data.drop(columns='Unnamed: 0', inplace=True)

        # Preserving original cleaned data and y_test not scaled
        # in different variables
        self.original_data = self.data.copy()
        self.y_test_notscaled = np.resize(np.array(self.data.tail(self.memory)['price_usd']), (self.memory, 1))

        # Fit a minmaxscaler on the target to recover it after the prediction
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(self.data[['price_usd']])

        # Scale all the data dataset
        global_scaler = MinMaxScaler()
        self.data[self.data.columns] = global_scaler.fit_transform(self.data[self.data.columns])

        # Extract y_test
        self.y_test = np.resize(np.array(self.data.tail(self.memory)['price_usd']), (self.memory, 1))

        #Extract X_for_prediction
        self.X_for_prediction = self.data.tail(self.memory).values

        # Extract X_test
        self.X_test = self.data[-self.memory*2:-self.memory].values

        # Remove X_test and y_test from Train Data
        self.data = self.data[:-self.memory*2]

        # Split data into X_train and X_val
        self.pre_X_train = self.data[:-self.memory*10]
        self.pre_X_val = self.data[-self.memory*10:]

    def subsample_sequence(self, length):
        '''
        function that return a random slice of features and targets
        len(X) = lenght and len(y) = 3
        '''
        last_possible = self.data.shape[0] - self.memory*2
        random_start = np.random.randint(0, last_possible)
        X = np.array(self.data.iloc[random_start: random_start+self.memory])
        y = np.array(self.data.iloc[random_start+self.memory:random_start+self.memory*2]['price_usd'])
        return X, y

    def subsample_sequence_train(self, length):
        '''
        function that return a random slice of features and targets
        len(X) = lenght and len(y) = 3
        '''
        last_possible = self.pre_X_train.shape[0] - self.memory*2
        random_start = np.random.randint(0, last_possible)
        X = np.array(self.pre_X_train.iloc[random_start: random_start+self.memory])
        y = np.array(self.pre_X_train.iloc[random_start+self.memory:random_start+self.memory*2]['price_usd'])
        return X, y

    def subsample_sequence_val(self, length):
        '''
        function that return a random slice of features and targets
        len(X) = lenght and len(y) = 3
        '''
        last_possible = self.pre_X_val.shape[0] - self.memory*2
        random_start = np.random.randint(0, last_possible)
        X = np.array(self.pre_X_val.iloc[random_start: random_start+self.memory])
        y = np.array(self.pre_X_val.iloc[random_start+self.memory:random_start+self.memory*2]['price_usd'])
        return X, y

    def get_X_y(self, length_of_observations):
        '''
        function that returns a list of random slices of features and targets
        len(X[0]) = lenght and len(y[0]) = 3
        '''
        X, y = [], []
        for length in length_of_observations:
            xi, yi = self.subsample_sequence(length)
            X.append(xi)
            y.append(yi)
        return X, y

    def get_X_y_train(self, length_of_observations):
        '''
        function that returns a list of random slices of features and targets
        len(X[0]) = lenght and len(y[0]) = 3
        '''
        X, y = [], []
        for length in length_of_observations:
            xi, yi = self.subsample_sequence_train(length)
            X.append(xi)
            y.append(yi)
        return X, y

    def get_X_y_val(self, length_of_observations):
        '''
        function that returns a list of random slices of features and targets
        len(X[0]) = lenght and len(y[0]) = 3
        '''
        X, y = [], []
        for length in length_of_observations:
            xi, yi = self.subsample_sequence_val(length)
            X.append(xi)
            y.append(yi)
        return X, y


    def extract_xy_tr_val(self):
        '''
        function returns a serie of train and test data
        train splits is the number of selections of our dataset
        train_time_min is the minimum number of days that are randomly choosen by the get_X_y function
        train_time_max is the maximum number of days that are randomly choosen by the get_X_y function
        '''
        length_of_observations = np.array([self.memory]*self.train_splits)
        self.X_train, self.y_train = self.get_X_y_train(length_of_observations)
        self.X_val, self.y_val = self.get_X_y_val(length_of_observations)

    def padding_seq(self):
        '''
        function that return the padded version of the train dataset
        to uniform the size of the model imput
        '''
        # Training dataset
        self.X_train_pad = pad_sequences(
            self.X_train,
            dtype='float32',
            value=-1)
        self.y_train_pad = pad_sequences(
            self.y_train,
            dtype='float32',
            value=-1).reshape(self.train_splits,
                              self.memory,
                              1)
        # Validation dataset
        self.X_val_pad = pad_sequences(
            self.X_train,
            dtype='float32',
            value=-1)
        self.y_val_pad = pad_sequences(
            self.y_train,
            dtype='float32',
            value=-1).reshape(self.train_splits,
                              self.memory,
                              1)

        # Test dataset
        self.X_test_pad = pad_sequences(
            self.X_test,
            dtype='float32',
            value=-1).reshape(1, self.memory, self.X_test.shape[1])

        self.X_for_prediction_pad = pad_sequences(
            self.X_for_prediction,
            dtype='float32',
            value=-1).reshape(1, self.memory, self.X_test.shape[1])

    def baseline_model(self):
        '''
        function that return a trained baseline model and its fitting history
        and save locally the trained model file basemodel.joblib
        '''
        clear_session()
        # normalizer = Normalization()
        # normalizer.adapt(self.X_train_pad)
        self.model = Sequential()
        # self.model.add(normalizer)
        # self.model.add(layers.Masking(mask_value=-1))
        self.model.add(layers.LSTM(self.rnn_layer1,
                                  activation='tanh',
                                  input_shape=(self.X_train_pad.shape[1],
                                      self.X_train_pad.shape[2]),
                                  return_sequences=True)) # GS N UNITS
        self.model.add(layers.Dense(self.dense_layer1, activation='relu')) # GS N UNITS
        self.model.add(layers.Dense(self.dense_layer2, activation='relu'))
        self.model.add(layers.Dense(self.dense_output, activation='linear'))
        self.model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.compile_learning_rate), metrics=[RootMeanSquaredError()])

        es = callbacks.EarlyStopping(patience=self.es_patience, restore_best_weights=True)

        self.history = self.model.fit(self.X_train_pad,
                    self.y_train_pad,
                    epochs=self.fit_epochs,
                    batch_size=self.fit_bach_size,
                    #validation_split=self.fit_validation_split,
                    validation_data=(self.X_val_pad, self.y_val_pad),
                    callbacks=[es],
                    verbose=1)
        joblib.dump(self.model, f'model_{date}.joblib')
        # joblib.dump(self.history, f'history_{date}.joblib')

        # Log the memory lenght decided
        self.mlflow_log_param("memory", self.memory)

        # MlFlow parameters log
        self.mlflow_log_param("extract_xy_tr_te_train_splits", self.train_splits)
        self.mlflow_log_param("padding", f'no padding, {self.memory} days')
        self.mlflow_log_param("rnn_layer1", self.rnn_layer1)
        self.mlflow_log_param("dense_layer1", self.dense_layer1)
        self.mlflow_log_param("dense_layer2", self.dense_layer2)
        self.mlflow_log_param("dense_output", self.dense_output)
        self.mlflow_log_param("compile_optimizer", self.compile_optimizer)
        self.mlflow_log_param("compile_loss", self.compile_loss)
        self.mlflow_log_param("compile_learning_rate", self.compile_learning_rate)
        self.mlflow_log_param("es_patience", self.es_patience)
        self.mlflow_log_param("fit_epochs", self.fit_epochs)
        self.mlflow_log_param("fit_bach_size", self.fit_bach_size)
        self.mlflow_log_param("fit_validation_split", f'{self.memory * 10} days of validation')
        self.mlflow_log_param("Architecture",
                              f'{self.memory} days input - {self.memory} days output - no overlapping - data_v2')
        self.mlflow_log_param("model_name", f'model_{date}.joblib')
        self.mlflow_log_param("n. of features", f'{self.X_train_pad.shape[2]}')

    def plot_history(self, title='', axs=None, exp_name=""):
        '''
        return the loss and metric plots of train and test fit process
        '''
        if axs is not None:
            ax1, ax2 = axs
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if len(exp_name) > 0 and exp_name[0] != '_':
            exp_name = '_' + exp_name
        ax1.plot(self.history.history['loss'], label = 'train' + exp_name)
        ax1.plot(self.history.history['val_loss'], label = 'val' + exp_name)
        ax1.set_ylim(0., 0.002)
        ax1.set_title('loss')
        ax1.legend()

        ax2.plot(self.history.history['mean_squared_error'],
                 label='train mse'  + exp_name)
        ax2.plot(self.history.history['val_mean_squared_error'],
                 label='val mse'  + exp_name)
        ax2.set_ylim(0., 0.002)
        ax2.set_title('mse')
        ax2.legend()

        return (ax1, ax2)

    def pred_3d_price(self):
        '''
        return the prediction of three days after the test data
        '''
        # extract only the first tree days of the prediction and test
        self.y_test_3d = self.y_test[0:3]
        self.y_pred_3d = np.array(self.model.predict(self.X_test_pad))[0,0:3]
        # extract the entire prediction
        self.y_pred = np.array(self.model.predict(self.X_test_pad))[0,:]
        # undo minmaxscaling on y_test and y_pred to have the real metric error
        self.real_y_pred_3d = self.target_scaler.inverse_transform(self.y_pred_3d)
        self.real_y_pred = self.target_scaler.inverse_transform(self.y_pred)
        self.real_y_test_3d = self.target_scaler.inverse_transform(self.y_test_3d)
        self.real_y_test = self.target_scaler.inverse_transform(self.y_test)
        self.rmse = (((self.real_y_test - self.real_y_pred)**2).mean())**0.5
        self.rmse_3d = (((self.real_y_test_3d - self.real_y_pred_3d)**2).mean())**0.5
        # Create  a new metric based on the last val_metric of the fit in dollar error
        self.val_rmse = self.target_scaler.inverse_transform(np.reshape(
            #self.history.history['val_mean_squared_error'][-1]**0.5, (1, -1)))[0,0]
            self.history.history['val_root_mean_squared_error'][-1], (1, -1)))[0,0]
        # MlFlow logging
        self.mlflow_log_metric('rmse', self.rmse)
        self.mlflow_log_metric('val_rmse', self.val_rmse)
        self.mlflow_log_metric('rmse_3d', self.rmse_3d)
        print(f'RMSE error: {self.rmse}')
        print(f'val RMSE error: {self.val_rmse}')
        print(f'RMSE error 3d: {self.rmse_3d}')

    def print_pred(self):
        '''
        plot the prediction on the real data
        '''
        fig = plt.figure(figsize=(20, 8))
        ax3 = fig.add_subplot()
        ax3.plot(self.target_scaler.inverse_transform(np.array(self.X_test_pad)[0,:,0].reshape(-1,1)))
        ax3.plot(range(self.memory, self.memory*2), self.target_scaler.inverse_transform(np.array(self.model.predict(self.X_test_pad))[0,:]))
        ax3.plot(range(self.memory, self.memory*2), self.y_test_notscaled)
        #ax3.set_xlim(60, 67)
        #ax3.set_ylim(self.y_test_notscaled[-1][0] - self.rmse, self.y_test_notscaled[-1][0] + self.rmse)
        ax3.set_title('pred vs real')
        ax3.legend()
        return ax3

    def get_prediction(self):
        '''
        function to retrive predicted data from the model
        '''
        self.model_a = joblib.load('model.joblib')
        self.prediction = np.array(self.model_a.predict(self.X_for_prediction_pad))[0,0:3]
        self.prediction = self.target_scaler.inverse_transform(self.prediction)
        self.real_y_test = self.target_scaler.inverse_transform(self.y_test)
        return self.prediction

    # PARAMETERS FOR GCP BASEMODEL UPLOAD

    def upload_model_to_gcp(self):
        '''
        function that upload the trained model to gcp
        '''
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename(f'model_{date}.joblib')
        print('model saved on gcp storage')
        # client_hst = storage.Client()
        # bucket_hst = client_hst.bucket(BUCKET_NAME)
        # blob_hst = bucket_hst.blob(STORAGE_LOCATION)
        # blob_hst.upload_from_filename(f'/tmp/history_{date}.joblib')
        # print('history saved on gcp storage')

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


if __name__ == '__main__':

    # download the data.csv from gc storage to test the code in gc ai

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob("data/data_26.csv")
    blob.download_to_filename("/tmp/data.csv")
    data = pd.read_csv('/tmp/data.csv') # TO RUN ON COLAB

    # TO RUN LOCALLY COMMENT ALL THE CODE BEHIND end DECOMMENT THE ROW BELOW
    #data = pd.read_csv('raw_data/data_26.csv')

    trainer = Trainer(data)
    trainer.preproc_data()
    trainer.extract_xy_tr_val()
    trainer.padding_seq()
    trainer.baseline_model()
    trainer.pred_3d_price()
    trainer.upload_model_to_gcp()
    #trainer.get_prediction()
