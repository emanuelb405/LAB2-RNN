
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
X = pd.read_csv('X.csv', sep=',',header=None)
Y = pd.read_csv('Y.csv', sep=',',header=None)
Y = Y.values


scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

X = np.reshape(X,(X.shape[0],X.shape[1],1))

number_of_samples = X.shape[1]
### Create Model"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM 
lstm = Sequential()
lstm.add(LSTM(units=100,return_sequences = True, input_shape = (number_of_samples,1)))
lstm.add(LSTM(units=100,return_sequences = False))
lstm.add(Dense(units=1))
lstm.compile(optimizer='adam',metrics=['mse'],loss='mean_squared_error')

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    './lstm2_1/base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)

csvlogger = CSVLogger(
    filename= "./lstm2_1/training_csv.log",
    separator = ",",
    append = False
)
callbacks = [checkpoint,csvlogger]

"""### Fit Model"""

lstm.fit(X,Y,epochs=100,batch_size=32,callbacks = callbacks,validation_split =0.33)

