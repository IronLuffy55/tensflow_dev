import fileinput
import time
import sys
import os 
import numpy as np
import json

input = []
for line in fileinput.input():
  print("Raw Line: ", line)
  # print("Line>", line, type(line), json.loads(line))
  input = json.loads(line)
  #np array
  input = np.asarray(input)
  #remove any value > max value of 20k
  input = input[(input < 20000)]
  print("Filtered Line: ",input)
  #buffer array to 80 elements
  from keras.preprocessing import sequence
  fixedinput = sequence.pad_sequences([input], maxlen=80)
  input = fixedinput[0]
  print("FI> " , fixedinput)
  print("I> ", input)
  # if input.size < 80:
  #   diff = 80 - input.size
  #   zeros = np.ndarray(shape=(1,diff), dtype=np.int32)
  #   zeros.fill(0)
  #   print(zeros)
  #   print(input)
  #   input = np.concatenate((zeros, input))
  # if input.size > 80:
  #   input = input[0:79]
print("Input type>", type(input))
print("Input length>", input.size)
print("Input max element>", np.argmax(input))
print("Input>", input)

import keras
from keras.models import load_model
from keras import backend as K

#print("Predict>", input)
model_file = "imdb_model.h5"
if not os.path.isfile(model_file):
  print("Model file does not exist>", model_file)
  exit("Dying")
  
model = load_model(model_file)
out = model.predict(fixedinput, 1, 1)
print("PRedictions>", out)

