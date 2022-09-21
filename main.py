from keras.layers import LSTM, Dropout, Dense, GRU
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10


def plot_prediction(test_result, prediction, company='HSI'):
    '''
    :param test_result: real values fortesting
    :param prediction: predicted values
    :return: print figure 
    '''

    plt.plot(test_result, color='red', label=f'Actual {company} Price')
    plt.plot(prediction, color='green', label=f'Predicted {company} Price')
    plt.title("HSI Stock Price")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")

    plt.legend()
    plt.show()


# Importing numpy npzfiles
print("[FILES] Importing npzfiles...")
train_npz = np.load('train.npz')
test_npz = np.load('test.npz')
val_npz = np.load('val.npz')
print("[FILES] Import finished.")

# Extracting data from npzfiles of train.npz
print("[DATA] Extracting data from npzfiles...")
x_train = train_npz['x']  # feature
y_train = train_npz['y']  # label
location_train = train_npz['locations']
time_train = train_npz['times']

# Extracting data from npzfiles of test.npz
x_test = test_npz['x']
location_test = test_npz['locations']
time_test = test_npz['times']

# Extracting data from npzfiles of val.npz
x_val = val_npz['x']
y_val = val_npz['y']
location_val = val_npz['locations']
time_val = val_npz['times']
print("[DATA] Extraction finished.")

# Checking the dimension of the data from train.npz
print("[DATA] Checking the dimension of the data from train.npz...")
print(x_train.shape)
print(y_train.shape)
print(location_train.shape)
print(time_train.shape)

# Checking the dimension of the data from test.npz
print("[DATA] Checking the dimension of the data from test.npz...")
print(test_npz['x'].shape)
print(test_npz['locations'].shape)
print(test_npz['times'].shape)

# Checking the dimension of the data from val.npz
print("[DATA] Checking the dimension of the data from val.npz...")
print(val_npz['x'].shape)
print(val_npz['y'].shape)
print(val_npz['locations'].shape)
print(val_npz['times'].shape)


# Scaling training set
print("[DATA] Scaling training set...")
scaler = MinMaxScaler(feature_range=(0, 1))

x_train = x_train.reshape(576000, 49)
x_train = scaler.fit_transform(x_train)
x_train = x_train.reshape(72000, 8, 49)

y_train = y_train.reshape(-1, 1)
y_train = scaler.fit_transform(y_train)
y_train = y_train.reshape(72000)

print("[DATA] Scaling testing set...")
x_test = x_test.reshape(12800, 49)
x_test = scaler.fit_transform(x_test)
x_test = x_test.reshape(1600, 8, 49)

print("[DATA] Scaling validation set...")
x_val = x_val.reshape(144000, 49)
x_val = scaler.fit_transform(x_val)
x_val = x_val.reshape(18000, 8, 49)

y_val = y_val.reshape(-1, 1)
y_val = scaler.fit_transform(y_val)
y_val = y_val.reshape(18000)

print("[DATA] Scaling finished.")

# Building model: LSTM
print("[LSTM] Building model...")
model = Sequential()
model.add(LSTM(units=128, return_sequences=True))  # First Layer
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))  # Second Layer
model.add(Dropout(0.2))
model.add(LSTM(units=128))  # Third Layer
model.add(Dropout(0.2))
model.add(Dense(units=1, input_shape=(1,)))  # Dense Layer
print("[LSTM] Building model finished.")

# Training model: RMSPROP
print("[TRAIN] Training model LSTM + RMSPROP...")
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x_train, y_train, epochs=1, batch_size=32)

# Building model: GRU
print("[GRU] Building model...")
model_gru = Sequential()
model_gru.add(GRU(50, return_sequences=True, activation='tanh'))  # First Layer
model_gru.add(Dropout(0.2))
model_gru.add(GRU(50, activation='tanh'))   # Second Layer
model_gru.add(Dropout(0.2))
model_gru.add(Dense(1))  # Dense Layer
print("[GRU] Building model finished.")

# Training model: SGD
print("[TRAIN] Training model GRU + SGD...")
model_gru.compile(optimizer=SGD(learning_rate=0.01,
                  decay=1e-7, momentum=0.9), loss='mse')
model_gru.fit(x_train, y_train, epochs=1, batch_size=32)

print("[TRAIN] Training model finished.")

# Prediction: LSTM
predicted_val_LSTM = model.predict(x_test)
predicted_val_LSTM = scaler.inverse_transform(predicted_val_LSTM)
predicted_val_LSTM.shape


# Visualization: LSTM
print("[VISUAL] Visualization: LSTM...")
plt.plot(predicted_val_LSTM, color='green', label=f'Predicted Value')
plt.title("Traffic Pick-up Volume")
plt.xlabel("Time")
plt.ylabel("Pick-up Value")
plt.legend()
plt.show()


# Prediction: GRU
predicted_val_GRU = model_gru.predict(x_test)
predicted_val_GRU = scaler.inverse_transform(predicted_val_GRU)
predicted_val_GRU.shape


# Visualization: GRU
print("[VISUAL] Visualization: GRU...")
plt.plot(predicted_val_GRU, color='red', label=f'Predicted Value')
plt.title("Traffic Pick-up Volume")
plt.xlabel("Time")
plt.ylabel("Pick-up Value")
plt.legend()
plt.show()
