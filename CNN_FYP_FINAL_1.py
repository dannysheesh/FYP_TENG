#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.utils.data.dataloader as dataloader
#from torch.utils.data import DataLoader
#import torch.optim as optim
#import pandas as pd
#import os

#from torch.utils.data import Dataset

#from torch.utils.data import TensorDataset
#from torchvision import transforms
#from torchvision.datasets import MNIST
#import torchvision

#import matplotlib.pyplot as plt
#import time
#from IPython.display import clear_output
#from skimage import io

#https://www.youtube.com/watch?v=FnDj8Jdu-X0


# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from sklearn import preprocessing


# In[ ]:





# In[74]:


def read_ucr(filename):
    data_import = np.loadtxt(filename, delimiter = ',')
    # forman: Each row is a sample. The first item in each row is the label, the rest is the time-series data
    label = data_import[:, 0]
    data = data_import[:, 1:]
    return data, label    


# In[75]:



#load time-series data

training_dir = 'C:/Users/mango/OneDrive/Documents/Uni/FYP/PuTTY logs/testing/files_train_all.csv'
testing_dir = 'C:/Users/mango/OneDrive/Documents/Uni/FYP/PuTTY logs/testing/files_test_all.csv'

x_train, y_train = read_ucr(training_dir)
x_test, y_test = read_ucr(testing_dir)

#perform the scaling for the Relu activation function
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.transpose(x_train))
x_train = np.transpose(scaler.transform(np.transpose(x_train)))

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.transpose(x_test))
x_test = np.transpose(scaler.transform(np.transpose(x_test)))

##get required stats about the data
#length of the time series
ts_len = x_train.shape[1]
#the number of train and test
num_train = x_train.shape[0]
num_test = x_test.shape[0]
#number of classes
num_classes = len(np.unique(y_test))



# In[76]:



#labels

#adjust the labels so that they are 0 - whatever

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(num_classes-1)

#one-hot encode the labels

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)




# In[77]:


#skip this
#def plot_grid_timeseries():
    


# In[78]:



#batch size must be changed!!!
#batch_size = 25
batch_size = num_train

num_channels = 1
n_epochs = 1000

##original
##convolutional layers
##layer 1
##filter_size1 = 5
#filter_size1 = 10
#num_filters1 = 25


##layer 2
#filter_size2 = 5
#num_filters2 = 50


#convolutional layers
#layer 1
#filter_size1 = 5
filter_size1 = 30
num_filters1 = 10


#layer 2
filter_size2 = 10
num_filters2 = 30



# In[79]:


#####

#construct the convolutional layer

def conv_layer(value, num_input_channels, filter_size, num_filters, use_pooling=True):
    
    tf.compat.v1.disable_eager_execution()
    
    #shape of the filter-weights for the convolution
    #this format is determined by the tesnorflow api
    shape = [filter_size, num_input_channels, num_filters]

    #create new weights (or filters) with the given shape
    weights = new_weights(shape=shape)
    
    #create new biases, one for each filter
    biases = new_biases(length=num_filters)
    
    #create the tensorflow operation for the convolution
    #a 1d convolution is performaed on the time-series with stride 1
    #the padding is set to 'same' so that the ourput is the same size as the input
    layer = tf.compat.v1.nn.conv1d(value=value, filters=weights, stride=1, padding='SAME')
    
    #add the biases to the results of the convolution
    layer += biases
    
    #activation function
    #re ReLu activation function is used
    layer = tf.nn.relu(layer)
    
    #pooling is used to downsamlpe the time-series
    if use_pooling:
        #max pooling is used where a pool size of 2 is used
        layer = tf.compat.v1.layers.max_pooling1d(inputs=layer, pool_size=2, strides=2, padding='SAME')
        
    #return the resulting layer and the weights
    return layer, weights
        






# In[80]:


#construct the fully connected layer

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    
    #create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    
    #calculate the layer as the matrix multiplication of the input and weights, and the add the bias-values
    layer = tf.matmul(input, weights) + biases
    
    #use the specified activation function
    if use_relu:
        layer = tf.nn.relu(layer)
    
    #return the resulting layer
    return layer



# In[81]:


#flatten the output of the convolutional layer

def flatten_layer(layer):
    
    #get the shape of the input layer
    layer_shape = layer.get_shape()
    
    num_features = layer_shape[1:4].num_elements()
    
    #reshape the layer
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    #return both the flattened layer and the number of features
    return layer_flat, num_features
    
    


# In[82]:


#initialise

def new_weights(shape):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))  
    


# In[83]:


#construct the convolutional neural network

#set up the placeholder tensors
#input data
tf.compat.v1.disable_eager_execution()

#replace 'tf' with 'tf.compat.v1'

x = tf.compat.v1.placeholder(tf.float32, [None, ts_len])
x_train_shape = tf.reshape(x, [-1, ts_len, num_channels])

#labels
y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_class = tf.argmax(y_true, axis=1)





# In[84]:


#convolutional layers

#layer 1 - convolutional neural layer
layer_conv1, weights_conv1 =     conv_layer(value=x_train_shape,
               num_input_channels=1,
               filter_size=filter_size1,
               num_filters=num_filters1,
               use_pooling=True)


#layer 1 - convolutional neural layer
layer_conv2, weights_conv2 =     conv_layer(value=layer_conv1,
               num_input_channels=num_filters1,
               filter_size=filter_size2,
               num_filters=num_filters2,
               use_pooling=True)




# In[85]:


#other layers

#layer 3 - flatten the ourput from layer 2, the convolutional layer
layer_flat, num_features = flatten_layer(layer_conv2)

#layer 4 - fully connected layer
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=num_classes, use_relu=False)


# In[86]:


#softmax to perform classification

#layer 5 - softmax layer
y_pred = tf.nn.softmax(layer_fc1)

#get the predicted class from the softmax layer
y_pred_class = tf.argmax(y_pred, axis=1)


# In[87]:


#cost function

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1, labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
##optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimimze(cost)

#optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_class, y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[88]:


#construct the tensorflow graph

#get the tensorflow session
session = tf.compat.v1.Session()

#initialise the session variables
session.run(tf.compat.v1.global_variables_initializer())


# In[89]:


#function to construct the batches for each epoch


#split the training data into batches
def batch_construction(batch_size, x_train, y_train_one_hot, shuffle=True):
    
    num_train, ts_len = x_train.shape
    
    #shuffle the data if required
    if shuffle:
        
        
#        idx_shuffle = np.rng.choice(num_train, num_train)
        idx_shuffle = np.random.choice(num_train, num_train)
#        x_train = np.rng.choice[idx_shuffle,:]
        x_train = x_train[idx_shuffle,:]
        
    
        y_train_one_hot = y_train_one_hot[idx_shuffle]
    
    
    #split the batches in a list, one for the data, one for the lbels
    minibatch = []
    minibatch_labels = []
    i=0
    while i < num_train:
        #extract the batch data from the enture training set
        if i + batch_size <= num_train:
            minibatch.append(x_train[i:i+batch_size,:])
            minibatch_labels.append(y_train_one_hot[i:i+batch_size,:])
        else:
            #construct a batch with the final elements and a random selection of the previous ones
            num_remain = num_train - i
            final_batch = np.zeros([batch_size, num_classes])
            final_batch_labels = np.zeros([batch_size, num_classes])
            final_batch[:num_remain,:] = x_train[i:num_train,:]                              #error on this line
            #ValueError: could not broadcast input array from shape (10,76) into shape (10,2)
            
            final_batch_labels[:num_remain,:] = y_train_one_hot[i:num_train,:]
            
            #select the remaining part of the batch randomly
            num_required = batch_size - num_remain
            
#            idx = np.rng.choice(batch_size, num_required)
            idx = np.random.choice(batch_size, num_required)
            
            final_batch[num_remain:,:] = x_train[idx,:]
            final_batch_labels[num_remain:,:] = y_train_one_hot[idx,:]
            minibatch.append(final_batch)
            minibatch_labels.append(final_batch_labels)
        #increment i
        i += batch_size
    
    return zip(minibatch, minibatch_labels)


# In[90]:


#evaluate the epoch
i1 = 0;
acc_mat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for epoch in range(n_epochs):
    
    #put each batch through the convolutional neural network
    
    #construct the minibatches for this epoch
    
    ###ARE THESE BATCH SIZES CAUSING THE ERROR??
    
    #batch_size = 20
    batch_size = num_train
    
    minibatch = batch_construction(batch_size, x_train, y_train_one_hot)
    
    for c_batch in minibatch:
        
        #feed in the batch and updates the weights using backpropagation
        x_batch = c_batch[0]
        y_true_batch = c_batch[1]
        
        #put the batch into a dict with the proper names for placeholder variables in the tensorflow graph
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        
        #run the optimiser using this batch of trainingg data
        #tensorflow assigns the variables in feed_dict_train to the placeholder variables and then reuns the optimiser
        session.run(optimizer, feed_dict=feed_dict_train)
    
    #use all the training data to construct the accuract after this epoch is complete
    feed_dict_train = {x: x_train, y_true: y_train_one_hot}
    
    if epoch % 100 == 0:
        #calculate the accuracy of the training-set
        acc = session.run(accuracy, feed_dict=feed_dict_train)*100
        acc_mat[i1] = acc
        i1 += 1
        print('Epoch: %03d Training Accuracy: %1.2f' %(epoch+1, acc))
print('Epoch: %03d Final Training Accuracy: %1.2f' %(epoch+1, acc))






# In[91]:


feed_dict = {x: x_test.reshape(-1, ts_len),
            y_true: y_test}

#calculate the predicted class using tensorflow
class_pred = session.run(y_pred_class, feed_dict=feed_dict)


#correct_sum = (np.array(y_test == class_pred)).sum()
#print(class_pred)

#create a boolean array whether each image is correctly classified
#m, n = size(y_test)
#print(m)
#print(n)

#for i in range(1, len(y_test)):
        
#    if (y_test[i-1] == class_pred[i-1]):
#        correct_counter += 1

correct_sum = np.array(y_test == class_pred).sum()
#print(correct_vals)

#print('total correct: %1.2f' %(correct_number))

correct_sum = 0;
total_sum = 0;

for i in range(1, len(y_test)):
    total_sum += 1
    if (y_test[i, 1]-class_pred[i]) == 0:
        correct_sum += 1
    #print(y_test[i, 0]-class_pred[i])
    
##print(correct_sum/total_sum)

print(correct_sum/len(y_test))
    
print('Correct: %1.2f' %(correct_sum))
print('Total: %1.2f' %(total_sum))
    
#correct = (y_test == class_pred)


##acc = (float(correct_sum)/num_test)*100

##print('Statistics on the testing time-series data')
##print('Accuracy: %1.2f%%' %(acc))




#print()



print(y_test)
print(class_pred)


print(acc_mat)


# ###### calculate the number of correctly classified images
# #when summing a boolean array, false means 0 and true means 1
# correct_sum = correct.sum()
# #correct_sum = np.sum(correct)
# 
# #classification accuracy is the number of correctly classified images
# #divided by the total number of images in the test set
# acc = (float(correct_sum)/num_test)*100
# 
# print('Statistics on the testing time-series data')
# print('Accuracy: %1.2f%%' %(acc))
# 
# 
# #get the confusion matrix using sklearn
# cm = confusion_matrix(y_true=y_test,
#                      y_pred=class_pred)
# 
# #print the confusion matric as text
# print('Accuracy on class 0: Correct: %i \ %i' %(cm[0, 0],np.sum(cm[0,:])))
# print('Accuracy on class 1: Correct: %i \ %i' %(cm[1, 1],np.sum(cm[1,:])))
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




