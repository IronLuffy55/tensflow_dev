import time
import sys
import os 
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K


if len(sys.argv) < 3:
  print("Tool requires 2 arguments: offset and batch size")
  exit()

path = '/root/keras_scripts/read_npy/mnist.npz'
print("Read dir>" + path)
f = np.load(path)
modelName = 'test2.h5'
model = load_model(modelName)

#variables
offset = int(sys.argv[1])
size = int(sys.argv[2])
num_classes = 10
epochs = 12
batch_size = 128
img_rows, img_cols = 28, 28

print('Offset>', offset, type(offset))
print('Batch size>', size, type(size))
x_train = f['x_train']
y_train = f['y_train']

stop = offset + size
x_train = x_train[offset:stop]
y_train = y_train[offset:stop]
f.close()

if K.image_data_format() == 'channel_first':
  x_train = x_train.reshape((x_train.shape[0], 1, img_rows, img_cols))
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

print('x_train: shape>', x_train.shape)
print('y_train: shape>', y_train.shape)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          # validation_data=(x_test, y_test))



# modelName = modelName + "_trained"
print("[%s]: Saving model:"%(time.time()))
model.save(modelName + '.h5')
print("[%s]: End model:"%(time.time()))
# save as JSON
print("[%s]: Saving json:"%(time.time()))
json_string = model.to_json()
with open(modelName + ".json","w") as f:
  f.write(json_string)
print("[%s]: End  json"%(time.time()))
# save as YAML
print("[%s]: Saving  yaml:"%(time.time()))
yaml_string = model.to_yaml()
with open(modelName + ".yaml","w") as f:
  f.write(yaml_string)
print("[%s]: End  yaml"%(time.time()))
print("[%s]: End"%(time.time()))

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])