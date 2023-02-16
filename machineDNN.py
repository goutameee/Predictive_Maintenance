import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Load the data
Xtrain = np.load('Xtrain.npy')
Ytrain = np.load('Ytrain.npy')
Xval = np.load('Xval.npy')
Yval = np.load('Yval.npy')
Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')

# Define the DNN architecture
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the DNN
model.fit(Xtrain, Ytrain, epochs=100, batch_size=32, validation_data=(Xval, Yval))

# Validate the DNN
Ypred_val = model.predict(Xval)
mse_val = np.mean((Yval - Ypred_val)**2)

# Test the DNN
Ypred_test = model.predict(Xtest)
mse_test = np.mean((Ytest - Ypred_test)**2)
