

#Generating Training Data
#Data Generator Function: iterates throught the train file in chunksizes and chooses a random batch of batch_size from every chunk
#every input file has the chunk size of 
import pandas as pd
import numpy as np
def generate_arrays_from_file(sliding_window = 1500, chunksize = 150000, batch_size = 100, read_until = 528777114):
  for chunk in pd.read_csv('train/train.csv', chunksize=chunksize ,dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32},nrows=read_until):
    dataX = []
    datay = []

    for i in range(0,batch_size):
      rand = np.random.randint(0,len(chunk)-sliding_window)
      dataX.append(chunk['acoustic_data'].iloc[rand:rand+sliding_window])
      datay.append(chunk['time_to_failure'].iloc[rand+sliding_window])
    dataX = np.array(dataX)
    datay = np.array(datay)
    dataX = np.reshape(dataX,(batch_size,sliding_window,1))
    datay = np.reshape(datay,(batch_size,1))
    yield dataX,datay

#Generating Validation Data
#Data Generator Function: iterates throught the train file in chunksizes and chooses a random batch of batch_size from every chunk
#every input file has the chunk size of 
import pandas as pd
import numpy as np
def generate_arrays_from_file_validation(sliding_window = 1500, chunksize = 150000, batch_size = 100, read_from = 528777114):
  for chunk in pd.read_csv('train/train.csv', chunksize=chunksize ,dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32},skiprows=read_from):
    dataX = []
    datay = []
    chunk.columns = ['acoustic_data','time_to_failure']
    for i in range(0,batch_size):
      rand = np.random.randint(0,len(chunk)-sliding_window)
      dataX.append(chunk['acoustic_data'].iloc[rand:rand+sliding_window])
      datay.append(chunk['time_to_failure'].iloc[rand+sliding_window])
    dataX = np.array(dataX)
    datay = np.array(datay)
    dataX = np.reshape(dataX,(batch_size,sliding_window,1))
    datay = np.reshape(datay,(batch_size,1))
    yield dataX,datay





"""### Create Model"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM 
lstm = Sequential()
lstm.add(LSTM(units=100,return_sequences = True, input_shape = (1500,1)))
lstm.add(LSTM(units=100,return_sequences = False))
lstm.add(Dense(units=1))
lstm.compile(optimizer='adam',metrics=['mse'],loss='mean_squared_error')

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    './lstm/base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './lstm/logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "./lstm/training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]

"""### Fit Model"""

batch_size = 100
train_steps = 528777114/batch_size-1000
val_steps = (629145480-528777114)/batch_size-1000

lstm.fit_generator(generate_arrays_from_file(),
                   steps_per_epoch=train_steps,
                   epochs=50,
                   validation_data = generate_arrays_from_file_validation(),
                   validation_steps = val_steps,
                   callbacks = callbacks)
