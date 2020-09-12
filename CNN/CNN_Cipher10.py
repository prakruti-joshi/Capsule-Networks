import keras
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.datasets import mnist
from keras.datasets import cifar10

(train_x,train_y), (test_x,test_y) = cifar10.load_data();

print('Training data shape', train_x.shape, train_y.shape)
print('Testing data shape', test_x.shape, test_y.shape)

classes = np.unique(train_y)
nClasses = len(classes)

print('Total number of classes:', nClasses)
print('Classes: ', classes)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)        #subplot(2,1,1)
plt.imshow(train_x[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_y[0]))
plt.show()

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_x[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_y[0]))
plt.show()

"""""
Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1
3D matrices.

Keras requires an extra dimension in the end which correspond to channels.
MNIST images are gray scaled so it use only one channel.
For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.
"""""

#train_x = train_x.reshape(-1,32,32,1)
#test_x = test_x.reshape(-1,28,28,1)
#print(train_x.shape, test_x.shape)

"""""
The data right now is in an int8 format, so before you feed it into the network you need to convert its type to float32, 
and you also have to rescale the pixel values in range 0 - 1 inclusive. So let's do that!

"""

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.
test_x = test_x / 255.

"""""
Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]).
"""

train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(train_x)
print('Original label: ', train_y[0])
print('after conversation to one hot', train_y_one_hot[0])

train_x, valid_X, train_label, valid_label = train_test_split(train_x,train_y_one_hot, test_size= 0.2 )
print(train_x.shape, valid_X.shape, train_label.shape, valid_label.shape)


### TRAINING THE DATA

batch_size = 64
epochs=10
num_classes= 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(32,32,3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes, activation='softmax'))


###### COMPILING THE MODEL

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))