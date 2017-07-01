import time
print("before from: %s"%(time.time()))
from tensorflow.examples.tutorials.mnist import input_data

print("after from: %s"%(time.time()))
import tensorflow as tf
import numpy as np


print("before download: %s"%(time.time()))
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("after download: %s"%(time.time()))
#2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32, [None, 784])

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

#Each step of the loop, we get a "batch" of one hundred random data points from our training set
print("before training: %s"%(time.time()))
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("after training: %s"%(time.time()))

#Evaluating Our Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("before accuracy: %s"%(time.time()))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("after accuracy: %s"%(time.time()))
