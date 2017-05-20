import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

saver = tf.train.import_meta_graph('my_test_model-25000.meta')
graph = tf.get_default_graph()

sess = tf.Session()
#saver.restore(sess,tf.train.latest_checkpoint('./'))
saver.restore(sess, 'my_test_model-25000')

W_conv1 = graph.get_tensor_by_name('W_conv1:0')
b_conv1 = graph.get_tensor_by_name('b_conv1:0')
W_conv2 = graph.get_tensor_by_name('W_conv2:0')
b_conv2 = graph.get_tensor_by_name('b_conv2:0')
W_fc1 = graph.get_tensor_by_name('W_fc1:0')
b_fc1 = graph.get_tensor_by_name('b_fc1:0')
W_fc2 = graph.get_tensor_by_name('W_fc2:0')
b_fc2 = graph.get_tensor_by_name('b_fc2:0')
#print(sess.run(b_fc2))
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))

#sess.run(tf.global_variables_initializer())

#y = tf.matmul(x,W) + b


img2 = cv2.imread('one_1.jpg',0)
img2 = np.multiply(img2,1/256)
print(img2)
height, width= img2.shape
#print (height, width)
#Format for the Mul:0 Tensor
#img2= cv2.resize(img2,dsize=(1,28*28), interpolation = cv2.INTER_CUBIC)

height, width = img2.shape
#print (height, width)

img3 = img2.reshape(1,28*28)
#print(img3.shape)


x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
a = print(x_image)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2),b_fc2)

best = y_conv.eval(session=sess,feed_dict = {x:img3,keep_prob:1.0})
#print(best)
result = sess.run(tf.nn.softmax(best))
print(result)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#a = mnist.test.images[4]
#c=mnist.test.images[5]
#b = np.multiply(a.reshape(28,28),256)
#d = np.multiply(a.reshape(28,28),256)
#
##print(b)
#cv2.imwrite('img4.jpg',b)
#cv2.imwrite('img5.jpg',d)
#
#
##a = print(x_image.eval(session=sess, feed_dict = {x:img3}))
