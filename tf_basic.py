import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


def linear_function():
    np.random.seed(1)

    X=tf.constant(np.random.randn(3,1),name="X")
    W=tf.constant(np.random.randn(4,3),name="W")
    b=tf.constant(np.random.randn(4,1),name="b")
    Y=tf.add(tf.matmul(W,X),b)

    sess=tf.Session()
    result=sess.run(Y)

    sess.close()

    return result

print( "result = " + str(linear_function()))


def sigmoid(z):
    x=tf.placeholder(tf.float32,name="x")
    sigmoid=tf.sigmoid(x)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

def cost(logits,labels):
    z=tf.placeholder(tf.float32,name="z")
    y=tf.placeholder(tf.float32,name="y")

    cost=tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)

    sess=tf.Session()
    cost=sess.run(cost,feed_dict={z:logits,y:labels})
    sess.close()

    return cost



logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))







































