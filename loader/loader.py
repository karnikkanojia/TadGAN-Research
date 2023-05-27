import tensorflow as tf
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from mlprimitives.custom.timeseries_preprocessing import rolling_window_sequences
from orion.primitives.timeseries_preprocessing import slice_array_by_dims

import math

class TadGANDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=16, shuffle=True, window_size=60, target_size=1, step_size=1, target_column=0):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(data.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.X, self.index = data.value.values, data.timestamp.values
        self.X = self.X.reshape(-1, 1)
        imp = SimpleImputer()
        self.X = imp.fit_transform(self.X)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = scaler.fit_transform(self.X)
        self.X, self.X_index, self.y, self.y_index = rolling_window_sequences(
            self.X, self.index, window_size=window_size, target_size=target_size, step_size=step_size, target_column=target_column)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.X))
        batch_x = self.X[low:high]
        batch_y = self.y[low:high].reshape(-1, 1)
        return batch_x
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.X = self.X[self.indexes]
            self.y = self.y[self.indexes]
            self.X_index = self.X_index[self.indexes]
            self.y_index = self.y_index[self.indexes]

    def get_tfdataset(self):
        return tf.data.Dataset.from_generator(
            lambda: self,
            output_types=(tf.float32),
            output_shapes=((None, self.X.shape[1], self.X.shape[2]))
        )

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('540821.csv', usecols=['Date', 'No. of Trades'], parse_dates=['Date'])
    data.rename(columns={'No. of Trades': 'value', 'Date': 'timestamp'}, inplace=True)
    data.timestamp = (data.timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    data.head()
    data_loader = TadGANDataLoader(data, shuffle=True)
    # print(data_loader[0].shape)
    datagen = data_loader.get_tfdataset()
    for i in iter(datagen):
        print(i.shape)