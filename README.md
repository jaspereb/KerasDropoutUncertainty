# KerasDropoutUncertainty
A very short example of using dropout uncertainty from (https://github.com/yaringal/DropoutUncertaintyCaffeModels) with Keras and Tensorflow. Including how to do test time dropout with keras.

This code runs a dropout model on the C02 dataset (see Keeling et al 2004), similar experiment to the above link (but a very simple model) written with Keras for simplicity. Note that the test dataset progresses in time, so we expect the first datapoints to have small uncertainty and the last data points to be very uncertain. The results could definitely be improved with larger models and dropout on every layer. 

Download the data folder, set the data paths in C02_Train.py then run that python file.

Requirements:
-Keras
-Tensorflow
-h5py

Note: if using your own dataset ensure that it is properly cleaned. I ran into some hard to pin down dropout behaviour from having a NaN in the training set. This seems to be Keras/Tensorflow related. 
