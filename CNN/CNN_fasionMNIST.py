#from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

print('Training data shape: ', train_x.shape, train_y.shape)
print('Testing data shape: ', test_x.shape, test_y.shape)

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

train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)
print(train_x.shape, test_x.shape)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.
test_x = test_x / 255.

train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(train_x)
print('Original label: ', train_y[0])
print('afer conversation to one hot', train_y_one_hot[0])

train_x, valid_X, train_label, valid_label = train_test_split(train_x,train_y_one_hot, test_size= 0.2 )
print(train_x.shape, valid_X.shape, train_label.shape, valid_label.shape)

### TRAINING THE DATA

batch_size = 64
epochs=10
num_classes= 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
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
