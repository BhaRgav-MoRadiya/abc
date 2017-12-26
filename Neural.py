from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
dataset = np.loadtxt("Converted-2.csv", delimiter=",",dtype='float')
Y =dataset[:1066,2]
Y1=dataset[1066:,2]
X = dataset[:1066,[1]]
X1=dataset[1066:,[1]]

model = Sequential()
model.add(Dense(2,input_dim=1,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

fit1=model.fit(X,Y,epochs=600,batch_size=10,verbose=1)
scores = model.predict(X1,verbose=1)

import matplotlib.pylab as plt

plt.plot(Y1, color='blue',label='Original')
plt.plot(scores, color='red',label='predicted')
plt.legend(loc='best')
plt.show(block=False)