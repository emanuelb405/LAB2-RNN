import pandas as pd
import numpy as np
def data_reduction_generator(window_size = 100, number_of_samples = 1500, chunksize = 10000000):
  for chunk in pd.read_csv('train/train.csv', chunksize=chunksize ,dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32}):
    chunk = chunk[chunk['acoustic_data']<=1000]
    print(len(chunk))
    x = []
    y = []
    
    #get the mean of certain sample size to reduce amount of data
    for i in range(0,(len(chunk)-len(chunk)%window_size)-window_size,window_size):
      a = chunk['acoustic_data'].iloc[i:i+window_size].mean()
      b = chunk['time_to_failure'].iloc[i+(int(window_size/2))]
      x.append(a)
      y.append(b)
      
    x_train = []
    y_train = []

    #get random samples for training from the dataset
    for k in range(0,number_of_samples):
      rand = np.random.randint(0,len(x)-number_of_samples)
      x_train.append(x[rand:rand+number_of_samples])
      y_train.append(y[rand+number_of_samples])
    
    x_train,y_train = np.array(x_train),np.array(y_train)

number_of_samples = 100
X = np.array(np.ones((1,number_of_samples)))
Y = np.array(np.ones((1)))
for x,y in data_reduction_generator(number_of_samples=number_of_samples):
  X = np.concatenate((X,x),axis=0)
  Y = np.concatenate((Y,y),axis=0)

np.savetxt("X.csv", X, delimiter=",")
np.savetxt("Y.csv", Y, delimiter=",")