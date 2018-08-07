import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras import backend as K
import h5py
import matplotlib as mp
import pickle

#Load the data
filename = 'PATH TO DATASET/train.h5'
f = h5py.File(filename, 'r')
# List all groups
print("Keys: %s" % f.keys())
train_key = list(f.keys())[0]
label_key = list(f.keys())[1]
# Get the data
train_data = np.asarray(list(f[train_key]))
train_labels = np.asarray(list(f[label_key]))

filename = 'PATH TO DATASET/test.h5'
f = h5py.File(filename, 'r')
# List all groups
print("Keys: %s" % f.keys())
test_key = list(f.keys())[0]
label_key = list(f.keys())[1]
# Get the data
test_data = np.asarray(list(f[test_key]))
test_labels = np.asarray(list(f[label_key]))

#Examine some data
print("Training set: {}".format(train_data.shape))
print("Testing set: {}".format(test_data.shape))

print(train_data[0:10])
print(train_labels[0:10])

#Define a model with dropout
def build_model():
  model = Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

EPOCHS = 200
model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=2)

#This is so that dropout runs at test time
getStochasticOutput = K.function([model.layers[0].input, K.learning_phase()], [model.layers[3].output])

#Run the network multiple times and determine the std deviation of these as an approximation of true variance
preds = np.array([])
certainty = np.array([])
mean = np.array([])

for step in range(0,np.size(test_data)):
    for n in range(0,50):
        #The learning phase flag must be 1
        preds = np.append(preds, getStochasticOutput([test_data[step:step+1], 1])[0])
        #print(predsNorm[n])
    print("Mean is:   {}".format(np.mean(preds)))
    print("Std at time {} is:   {}".format(test_data[step], np.std(preds)))
    certainty = np.append(certainty, np.std(preds))
    mean = np.append(mean, np.mean(preds))

with open('results.pkl', 'w') as resultsFile:
    pickle.dump([certainty, mean, test_data], resultsFile)

#Open results.pkl in ipython and run:
#import matplotlib as mp
#mp.pyplot.errorbar(test_data, mean, certainty, errorevery=15, ecolor='r')

