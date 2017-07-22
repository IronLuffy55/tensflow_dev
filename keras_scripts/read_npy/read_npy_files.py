# import keras 
import numpy as np
from keras import backend as K

#path = '/root/keras_scripts/read_npy/mnist/x_train.npy'
#read mnist data set
path = '/root/keras_scripts/read_npy/mnist.npz'
print("Read dir>" + path)
f = np.load(path)

x_train = f['x_train']

# y_train = f['y_train']
# x_test = f['x_test']
# y_test = f['y_test']
f.close()
np.set_printoptions(linewidth=200)
# print("x_train:\n",x_train[0])
# print("y_train:\n",y_train[0])
# print("x_test:\n",x_test[0])
# print("y_test:\n",y_test[0])

num_classes = 10
# y_train = keras.utils.to_categorical(y_train, num_classes)
# print("Categorical y_train:\n",y_train[0])

# y_test = keras.utils.to_categorical(y_test, num_classes)
# print("Categorical y_test:\n",y_test[0])

##Printing outout of categorical
# count = 10
# counter = 0
# for num in y_train:
#   if counter > count:
#     break;
#   print("y number: ",num)
#   print('y onehot: ', keras.utils.to_categorical(num, num_classes)[0])
#   print('----------------------------------------------------')
#   counter = counter + 1

##testing tuple math
# print('X_train shape:', x_train.shape)


# if K.image_data_format() == 'channel_first':
#   input_shape = (1, img_rows, img_cols)
# else:
#   input_shape = (img_rows, img_cols, 1)


# x_train = x_train.reshape((x_train.shape[0], ) + input_shape)
# x_test = x_test.reshape((x_test.shape[0], ) + input_shape)

##testing file write
# estr = "Ello!"
# with open("test.txt","w") as f:
#   f.write(estr)

#slice aarray
small_x_train = x_train[0:1000]
print('small x_train shape:', small_x_train.shape)
print('small x_train:', small_x_train[0])