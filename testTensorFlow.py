import numpy as np
import tensorflow as tf

a1 = tf.ones([10])
sess = tf.Session()

hello = tf.constant('Hello, Joe!')
print(sess.run(hello))
print(hello)

a = tf.constant(10)
b = tf.constant(20)
c = a + b
print(sess.run(c))

with tf.Session():
    a = tf.constant(10)
    b = tf.constant(20)
    c = a + b
    print c
    print(c.eval())

# Exercise : Print out the sigmoid value of 2x2 matrix of 1's, using TensorFlow

shape=[2,2]
a2 = tf.ones(shape)
with tf.Session ():
    a3=(tf.sigmoid(a2))
    print a3.eval()

shape=[2,2]
a1=tf.ones(shape)
with tf.Session ():
    a2=(tf.sigmoid(a1))
    print a2.eval()
    
