import numpy as np
import tensorflow as tf

#model coefficients
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#model input and outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#linear model
linear_model = W * x + b

#loss function (sum of squares)
loss = tf.reduce_sum(tf.square(linear_model - y)) 

#optimizer (reduces loss values i.e. error value)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4] #what my given inputs are
y_train = [0,-1,-2,-3] #what my outputs should be 

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) #reset values

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

print(sess.run([W, b]))


#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

