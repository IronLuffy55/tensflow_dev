import time
print("before from: %s"%(time.time()))
from tensorflow.examples.tutorials.mnist import input_data

print("after from: %s"%(time.time()))
import tensorflow as tf
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784])

#First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 

print("before download: %s"%(time.time()))
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("after download: %s"%(time.time()))


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

#tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1]
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
print("before interactive session start: %s"%(time.time()))

sess = tf.InteractiveSession()
print("after interactive session start: %s"%(time.time()))

print("before gbi run: %s"%(time.time()))
tf.global_variables_initializer().run()
print("after gbi run: %s"%(time.time()))
saver = tf.train.Saver()

#Each step of the loop, we get a "batch" of one hundred random data points from our training set
print("before training: %s"%(time.time()))
for _ in range(10):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("after training: %s"%(time.time()))

#Evaluating Our Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("before accuracy: %s"%(time.time()))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("after accuracy: %s"%(time.time()))
save_path = saver.save(sess, "./tensor_flow_models/simple_mnist_model.ckpt")
print("Model saved in file: %s" % save_path)
