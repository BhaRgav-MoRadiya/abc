from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import numpy
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
numpy.random.seed(7)
#dataset = np.loadtxt("Full-data.csv", delimiter=",",dtype='float')

import pandas as pd

Newdataset=pd.read_csv('Full-data.csv')
#Newdataset=Newdataset.dropna(how='any',axis=0) 
#Newdataset.mask(Newdataset.astype(object).eq('None')).dropna()
Newdataset = Newdataset.replace(to_replace='None', value=np.nan).dropna()
Newdataset = Newdataset.values

#print (Newdataset.head())
Y=Newdataset[:,0]
encoder = LabelEncoder()
encoder.fit(Y)
mod_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(mod_Y)

dataset=Newdataset[:,1:].astype(float)
dataset=np.append(dataset,dummy_y,axis=1)
#print(dataset)
#a = numpy.asarray(dataset)
#numpy.savetxt("foo.csv", a, delimiter=",")




#Y =dataset[:,2]
#Y1=dataset[143000:,4]
#X =dataset[:,[0,1,5,6,7,8,9,10,11,12,13,14,15]]
#X1=dataset[143000:,[0,1,2,3]]

from sklearn.preprocessing import StandardScaler

# Load data and split into testing and training data

scale = StandardScaler()
scale.fit(dataset[:,[1,2,5,6]])
new_training_data = scale.transform(dataset[:,[1,2,5,6]])
new_training_data=np.append(dataset[:,[0,7,8,9,10,11,12,13,14,15]],new_training_data,axis=1)
print(new_training_data)

X=new_training_data[:,[0,1,2,3,4,5,6,7,8,9,10,12,13]]
Y=new_training_data[:,11]
model = Sequential()
model.add(Dense(13,input_dim=13,kernel_initializer='glorot_normal'))
model.add(Dense(260,activation='relu'))
#model.add(Dense(26,activation='relu'))
model.add(Dense(130,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')
fit1=model.fit(X,Y,epochs=400,batch_size=100,verbose=1,validation_split=0.1)
#fit2=model.evaluate(X1, Y1, batch_size=50, verbose=1)
#scores = model.predict(X1,verbose=1)
'''
import matplotlib.pylab as plt
#plt.plot(Y1, color='blue',label='Original')
#plt.plot(scores, color='red',label='predicted')
#plt.legend(loc='best')
#plt.show(block=False)
#print('Accuracy %f'% (fit2[1]))
plt.plot(new_training_data[:,0], color='blue',label='Training Loss')
plt.plot(new_training_data[:,1], color='red',label='Training Loss')
plt.plot(new_training_data[:,2], color='green',label='Training Loss')
plt.plot(new_training_data[:,3], color='black',label='Training Loss')
#plt.plot(fit2[0],color='red',label='Evaluation loss')
plt.legend(loc='best')
plt.show(block=False)
'''
