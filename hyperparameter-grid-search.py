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
#print(new_training_data)

X=new_training_data[:,[0,1,2,3,4,5,6,7,8,9,10,12,13]]
Y=new_training_data[:,11]






# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(13,input_dim=13,kernel_initializer='glorot_normal'))
    model.add(Dense(60,activation='relu'))
    model.add(Dense(40,activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model
# create model
model = KerasClassifier(build_fn=create_model, verbose=1)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y, verbose=1)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))