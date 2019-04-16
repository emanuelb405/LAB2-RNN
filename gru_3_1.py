import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
X = pd.read_csv('X.csv', sep=',',header=None)
Y = pd.read_csv('Y.csv', sep=',',header=None)
Y = Y.values


scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

X = np.reshape(X,(X.shape[0],X.shape[1],1))

number_of_samples = X.shape[1]

# Create Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU 
gru = Sequential()
gru.add(GRU(units=100,return_sequences = True, input_shape = (number_of_samples,1)))
gru.add(Dropout(0.3))
gru.add(GRU(units=100,return_sequences = False))
gru.add(Dropout(0.3))
gru.add(Dense(units=1))
gru.compile(optimizer='adam',metrics=['mse'],loss='mean_squared_error')

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    './gru3_1/base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)


csvlogger = CSVLogger(
    filename= "./gru3_1/training_csv.log",
    separator = ",",
    append = False
)



callbacks = [checkpoint, csvlogger]

"""### Fit Model"""

gru.fit(X,Y,epochs=100,batch_size=32,callbacks = callbacks,validation_split =0.33)

