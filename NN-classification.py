from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import numpy
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#dataset = np.loadtxt("Full-data.csv", delimiter=",",dtype='float')

import pandas as pd

Newdataset=pd.read_csv('NN-classification.csv')
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

Y1=Newdataset[:,1]
encoder = LabelEncoder()
encoder.fit(Y1)
mod_Y1 = encoder.transform(Y1)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y1 = to_categorical(mod_Y1)



dataset=Newdataset[:,2:].astype(float)
dataset=np.append(dataset,dummy_y1,axis=1)
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
scale.fit(dataset[:,[0,1,2,3,4,5,6]])
new_training_data = scale.transform(dataset[:,[0,1,2,3,4,5,6]])
new_training_data=np.append(new_training_data,dataset[:,7:],axis=1)
print(new_training_data)


X=new_training_data[:,:16]
Y=new_training_data[:,16:]

import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=16, activation='relu'))
	model.add(Dense(16, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=50, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X,Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


'''
model = Sequential()
model.add(Dense(32,input_dim=16,activation='relu'))
#model.add(Dense(64,activation='relu'))
#model.add(Dense(64,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
fit1=model.fit(X,Y,epochs=400,batch_size=32,verbose=1,validation_split=0.33)
#fit2=model.evaluate(X1, Y1, batch_size=50, verbose=1)
#scores = model.predict(X1,verbose=1)
'''
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