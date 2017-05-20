from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
#sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape,name_var):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = name_var)

def bias_variable(shape,bias_var):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=bias_var)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32],"W_conv1")
b_conv1 = bias_variable([32],"b_conv1")

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
b_conv2 = bias_variable([64],"b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
b_fc1 = bias_variable([1024],"b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10],"W_fc2")
b_fc2 = bias_variable([10],"b_fc2")

#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2),b_fc2,name="op_to_restore")
#y_conv = tf.add(tf.matmul(h_conv1, W_conv1),b_conv1,name="op_to_restore")

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for i in range(25000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print(sess.run(b_fc2))
  train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#save_path = saver.save(sess, '/home/rohit/python_work/git_dir/mnist_tut.ckpt')
#print (results)
saver.save(sess, 'my_test_model',global_step = 25000)
#saver.save(sess, 'my-model', global_step=100) ==> filename: 'my-model-100'
#a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(sess.run(b_fc2))
#print(sess.run(a))
