#this function loads part of the csv file and samples randomly from it
#it does this only for the first part of the whole csv file which is 
#the training set (up to earthquake 13)
import pandas as pd
import numpy as np
def generator_train(number_of_samples = 300,input_shape = 1500,
                           chunksize = 5000000,batch_size=32,read_until = 528777114):
  #number of samples: number of training samples to sample from a chunk and feed to the network in one iteration
  #chunksize: number of samples to read from the csv file
  #input_shape: input shape of the network and amount of input samples that are sent to the network
  while True:
    from sklearn.model_selection import train_test_split

    for chunk in pd.read_csv('train/train.csv', chunksize=chunksize ,
                             dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32},
                             nrows=read_until):

      chunk = chunk[chunk['acoustic_data']<=1000]

      print(len(chunk))

      x = chunk['acoustic_data']
      y = chunk['time_to_failure']

      x_train = []
      y_train = []

      #get random samples for training from the dataset with the amount of random
      for k in range(0,number_of_samples):
        rand = np.random.randint(0,len(chunk)-input_shape)
        x_train.append(x.iloc[rand:rand+input_shape])
        y_train.append(y.iloc[rand+input_shape])

      X_train,y_train = np.array(x_train),np.array(y_train)
      #X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

      for batch in range(0,len(X_train),batch_size):

        X_batch = X_train[batch:batch+batch_size,:]
        y_batch = y_train[batch:batch+batch_size]
        X_batch = np.reshape(X_batch,(X_batch.shape[0],X_batch.shape[1],1))
        print(X_batch.shape)
        yield X_batch, y_batch

#the folowing is the test function that reads the end of the file as a test and validation set (up from 13 to 16)
#this function loads part of the csv file in chunks


#this function loads part of the csv file and samples randomly from it and returns a train
#test split of the sampled data
import pandas as pd
import numpy as np
def generator_validation(number_of_samples = 300,input_shape = 1500, chunksize = 5000000,
                         batch_size=32,read_from = 528777114):
  #number of samples: number of training samples to sample from a chunk and feed to the network in one iteration
  #chunksize: number of samples to read from the csv file
  #input_shape: input shape of the network and amount of input samples that are sent to the network
  while True:
    

    for chunk in pd.read_csv('train/train.csv', chunksize=chunksize ,
                             dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32},
                             skiprows=read_from):
      col = chunk.columns
      chunk.rename(columns={col[0]: 'acoustic_data',col[1]:'time_to_failure'}, inplace=True)
      print(chunk.head())
      chunk = chunk[chunk['acoustic_data']<=1000]

      print(len(chunk))

      x = chunk['acoustic_data']
      y = chunk['time_to_failure']

      x_train = []
      y_train = []

      #get random samples for training from the dataset with the amount of random
      for k in range(0,number_of_samples):
        rand = np.random.randint(0,len(chunk)-input_shape)
        x_train.append(x.iloc[rand:rand+input_shape])
        y_train.append(y.iloc[rand+input_shape])

      X_test,y_test = np.array(x_train),np.array(y_train)
      #X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

      for batch in range(0,len(X_test),batch_size):
        X_batch = X_test[batch:batch+batch_size,:]
        y_batch = y_test[batch:batch+batch_size]
        X_batch = np.reshape(X_batch,(X_batch.shape[0],X_batch.shape[1],1))

        yield X_batch, y_batch


# it utilizes the model that was improved using previous work, although the data is not scaled (which this data allows looking at previous work)

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU 
gru = Sequential()
gru.add(GRU(units=100,return_sequences = True, input_shape = (1500,1)))
gru.add(Dropout(0.3))
gru.add(GRU(units=100,return_sequences = False))
gru.add(Dropout(0.3))
gru.add(Dense(units=1))
gru.compile(optimizer='adam',metrics=['mse'],loss='mean_squared_error')

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    './gru4_1/base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)


csvlogger = CSVLogger(
    filename= "./gru4_1/training_csv.log",
    separator = ",",
    append = False
)


callbacks = [checkpoint,csvlogger]


train_steps = 500
val_steps = 100


gru.fit_generator(generator_train(),
                   steps_per_epoch=train_steps,
                   epochs=50,
                   validation_data = generator_validation(),
                   validation_steps = val_steps,
                   callbacks = callbacks)
