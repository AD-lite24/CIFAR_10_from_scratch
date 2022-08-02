# %%
import numpy as np
import math
from sklearn.metrics import accuracy_score 
from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from keras.datasets import cifar10  #to import dataset

# %%
tf.test.gpu_device_name()
#testing for gpu

# %%
#If running in kaggle

# import os
# for dirname, _, filenames in os.walk('/kaggle/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

#else use this
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
#------------for kaggle again---------------#

# import numpy as np
# data = np.load("/kaggle/input/cifar10kerasfilescifar10loadingdata/cifar-10.npz")
# filenames = ["x_train","y_train","x_test","y_test"]
# nps = []
# for filename in filenames:
#     nps.append(data[filename])
# x_train,y_train,x_test,y_test = nps

#-------------------------------------------#

print(x_train.shape)
print(y_train.shape)

# %%
x_train = x_train.astype('float32')
x_train = x_train/255
x_test = x_test.astype('float32')
x_test = x_test/255

# %%
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]


# %%
class CifarClass():

  def __init__(self, learning_rate = 0.01):
    
    self.training_images = None
    self.training_labels = None 
    self.testing_images = None
    self.testing_labels = None
    self.__learning_rate = learning_rate

  def setup_of_images(self):
    self.training_images = x_train
    train_len = len(self.training_images)
    self.training_labels = y_train

    self.testing_images = x_test
    test_len = len(self.testing_images)
    self.testing_labels = y_test

  def __ReLU(self, Z):
    return np.maximum(Z,0)

  def __ReLU_derivative(self, z):
    return z>0

  def __softmax_score(self, x):
    exponent = np.exp(x - np.max(x))
    return exponent/exponent.sum()

  def __forward_pass(self, X):
    
    self.__output1 = X.dot(self.__weights_1) + self.__bias_1      #shape: (1, 1024) 
    self.__output1_relu = self.__ReLU(self.__output1)               #shape: (1, 1024)
    self.__output2 = self.__output1_relu.dot(self.__weights_2) + self.__bias_2  #shape: (10, 1)
    self.__output2_softmax = self.__softmax_score(self.__output2)               #shape: (10, 1)
    

  def __backward_propogation(self, cross_entropy_gradient_2, difference_total_2, difference_total_1, cross_entropy_gradient_1):

    #updating weights_2
    step_size_2 = (cross_entropy_gradient_2*self.__learning_rate).T     #shape: (1024, 10)
    self.__weights_2 = self.__weights_2 - step_size_2

    #updating bias_2
    self.__bias_2 = self.__bias_2 - (difference_total_2)/len(self.__x_batch)    #(1,10)
    

    #updating weights_1
    step_size_1 = (cross_entropy_gradient_1*self.__learning_rate)     #(3072, 1024)
    self.__weights_1 = self.__weights_1 - step_size_1

    #updating bias_1
    self.__bias_1 = self.__bias_1 - (difference_total_1.T)/len(self.__x_batch)   #(1,1024)

  def train(self, X, y):

    self.__class_count = 10
    self.__weights_1 = np.random.rand(3072, 1024) - 0.5
    self.__weights_2 = np.random.rand(1024, 10) - 0.5
    self.__bias_1 = np.random.rand(1, 1024) - 0.5
    self.__bias_2 = np.random.rand(1,10) - 0.5
    
    print('initial b_2:', self.__bias_2)
    print('initial w_2:', self.__weights_2)

    self.__num_epochs = 25

    for i in range(self.__num_epochs):
      print('epoch idx:', i)
      for j in range(3125):
        
        self.__x_batch = X[j]                     #loading batches
        self.__y_batch = y[j]

        cross_entropy_sum_2 = 0
        cross_entropy_sum_1 = 0
        difference_total_1 = 0
        difference_total_2 = 0
        
        for k in range(16):                       #loading images for forward pass
          
          self.__forward_pass(self.__x_batch[k])   #k is image index(sanity check)
          cross_entropy_sum_2 += ((self.__output2_softmax - self.__y_batch[k]).T).dot(self.__output1_relu)  #shape: (10,1024)   
          difference_total_2 += self.__output2_softmax - self.__y_batch[k]  #(1,10)
          temp = self.__x_batch[k].reshape(3072,1)
          cross_entropy_sum_1 += (temp).dot((self.__weights_2.dot((self.__output2_softmax - self.__y_batch[k]).T).T)*self.__ReLU_derivative(self.__output1)) #(3072, 1024)
          difference_total_1 += self.__weights_2.dot((self.__output2_softmax - self.__y_batch[k]).T)            #(1024,1)

        cross_entropy_gradient_2 = cross_entropy_sum_2/len(self.__x_batch)
        cross_entropy_gradient_1 = cross_entropy_sum_1/len(self.__x_batch)

        #backward pass for the batch
        self.__backward_propogation(cross_entropy_gradient_2, difference_total_2, difference_total_1, cross_entropy_gradient_1)

      predict = self.predict(cc.testing_images)
      print('accuracy for epoch idx', i, 'is:', accuracy_score(cc.testing_labels, predict))

  def predict(self, X):
    y= np.zeros((len(X), self.__class_count))
    for i in range(len(X)):
      self.__forward_pass(X[i])
      max_score_idx = np.argmax(self.__output2_softmax)
      y[i][max_score_idx] = 1
    return y

  
  
    

# %%
cc = CifarClass() 
cc.setup_of_images()

print (cc.training_images.shape)
print(cc.testing_images.shape)


# %%
#flattening

cc.training_images = cc.training_images.reshape(50000, 3072)

cc.training_images = cc.training_images - 0.5     

print(cc.training_images.shape)
cc.training_images = cc.training_images.reshape(3125, 16, 3072)
print('final training image shape:', cc.training_images.shape)
cc.training_labels = cc.training_labels.reshape(3125, 16, 10)         
print('final training labels shape', cc.training_labels.shape)
cc.testing_images = cc.testing_images.reshape(10000, 3072)
cc.testing_images = cc.testing_images - 0.5
print('final testing images shape', cc.testing_images.shape)

print(cc.training_labels[23][15])

#more testing
temp = cc.training_labels[23]
print(temp[15].shape)
print(temp[15][2])

# %%
def main():
  X_train = cc.training_images
  y_train = cc.training_labels

  X_test = cc.testing_images
  y_test = cc.testing_labels

  cc.train(X_train, y_train)

  global predictions 
  predictions = cc.predict(X_test)

  print('final accuracy', accuracy_score(y_test, predictions))

# %%
if __name__ == "__main__":
    main()
    


