import time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import sys
print("Imports work just fine")

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print("Slicing training data set to 1000")
# x_train = x_train[0:4000]
# y_train = y_train[0:4000]
# x_test = x_test[0:4000]
# y_test = y_test[0:4000]
# print("Y_test:>",y_test)
# sys.exit("Dying")

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('beginning: x_train[0]>',x_train[0])
print('beginning: x_train>',x_train)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('after: x_train[0]>',x_train[0])
print('after: x_train>',x_train)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print("[%s]: Saving model: imdb_model.h5"%(time.time()))
model.save('imdb_model.h5')