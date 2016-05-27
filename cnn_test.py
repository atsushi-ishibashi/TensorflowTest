# -*- coding: utf-8 -*-

import wave_data
import tensorflow as tf
import csv

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#padding="SAME" : input size equal to output size
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    mnist = wave_data.read_data_sets()
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 4096])
    y_ = tf.placeholder("float", shape=[None, 2])
    keep_prob = tf.placeholder("float")

    x_image = tf.reshape(x, [-1,64,64,1])
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    W1_hist = tf.histogram_summary("conv1_weights",W_conv1)
    b1_hist = tf.histogram_summary("conv1_biases",b_conv1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    W2_hist = tf.histogram_summary("conv2_weights",W_conv2)
    b2_hist = tf.histogram_summary("conv2_biases",b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)#image:32*32, shape=[None, 32, 32, 64]

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    W3_hist = tf.histogram_summary("conv3_weights",W_conv3)
    b3_hist = tf.histogram_summary("conv3_biases",b_conv3)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)#image:16*16

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])
    W4_hist = tf.histogram_summary("conv4_weights",W_conv4)
    b4_hist = tf.histogram_summary("conv4_biases",b_conv4)
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)#image:8*8

    W_fc4 = weight_variable([8*8*256, 2048])
    b_fc4 = bias_variable([2048])
    W4_hist = tf.histogram_summary("fc4_weights",W_fc4)
    b4_hist = tf.histogram_summary("fc4_biases",b_fc4)
    h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*256])
    h_fc4 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc4) + b_fc4)
    keep_prob = tf.placeholder("float")
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

    W_fc5 = weight_variable([2048, 256])
    b_fc5 = bias_variable([256])
    W5_hist = tf.histogram_summary("fc5_weights",W_fc5)
    b5_hist = tf.histogram_summary("fc5_biases",b_fc5)
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)
    h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

    W_fc6 = weight_variable([256, 2])
    b_fc6 = bias_variable([2])
    W6_hist = tf.histogram_summary("fc6_weights",W_fc6)
    b6_hist = tf.histogram_summary("fc6_biases",b_fc6)
    y_conv=tf.nn.softmax(tf.matmul(h_fc5_drop, W_fc6) + b_fc6)

    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,100.0)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    merged = tf.merge_all_summaries()
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("./tmp/swt_test_0119",sess.graph_def)

    print "start training"
    for i in range(2001):
      if i%10 == 0:
          print "step %d"%i
          try:
              result = sess.run([merged])
              writer.add_summary(result[0],i)
          except Exception as e:
              print '=== エラー内容 ==='
              print 'type:' + str(type(e))
              print 'args:' + str(e.args)
              print 'message:' + e.message
              print 'e自身:' + str(e)
      batch = mnist.train.next_batch(20)
      if i%250 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
          print "step %d, training accuracy %g, test accuracy %g"%(i, train_accuracy, test_accuracy)
      elif i%50 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          print "step %d, training accuracy %g"%(i, train_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print "finish training"
    test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print "test accuracy %g"%(test_accuracy)


if __name__ == '__main__':
    main()
