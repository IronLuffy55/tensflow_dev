import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import os 

print('Arguments:', sys.argv)
num_classes = 10
modelName = ""

modelName = sys.argv[1]
print('Model Name: ', modelName)
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channel_first':
  input_shape = (1, img_rows, img_cols)
else:
  input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



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