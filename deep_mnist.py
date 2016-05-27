def softmax_test():
    import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    import tensorflow as tf
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.initialize_all_variables())

    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

def deep_cnn_test():
    import input_data
    import tensorflow as tf
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.initialize_all_variables())

    y = tf.nn.softmax(tf.matmul(x,W) + b)

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

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W1_hist = tf.histogram_summary("conv1_weights",W_conv1)
    b1_hist = tf.histogram_summary("conv1_biases",b_conv1)
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    W_conv2 = weight_variable([5, 5, 32, 32])
    b_conv2 = bias_variable([32])
    W2_hist = tf.histogram_summary("conv2_weights",W_conv2)
    b2_hist = tf.histogram_summary("conv2_biases",b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)#14*14

    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    W3_hist = tf.histogram_summary("conv3_weights",W_conv3)
    b3_hist = tf.histogram_summary("conv3_biases",b_conv3)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    W_conv4 = weight_variable([5, 5, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    W4_hist = tf.histogram_summary("conv4_weights",W_conv4)
    b4_hist = tf.histogram_summary("conv4_biases",b_conv4)

    W_conv5 = weight_variable([5, 5, 64, 64])
    b_conv5 = bias_variable([64])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    W5_hist = tf.histogram_summary("conv5_weights",W_conv5)
    b5_hist = tf.histogram_summary("conv5_biases",b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)#7*7

    W_fc6 = weight_variable([7*7*64, 1024])
    b_fc6 = bias_variable([1024])
    W6_hist = tf.histogram_summary("fc6_weights",W_fc6)
    b6_hist = tf.histogram_summary("fc6_biases",b_fc6)
    h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*64])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)
    keep_prob = tf.placeholder("float")
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    W_fc7 = weight_variable([1024, 1024])
    b_fc7 = bias_variable([1024])
    W7_hist = tf.histogram_summary("fc7_weights",W_fc7)
    b7_hist = tf.histogram_summary("fc7_biases",b_fc7)
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)
    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    W_fc8 = weight_variable([1024, 10])
    b_fc8 = bias_variable([10])
    W8_hist = tf.histogram_summary("fc8_weights",W_fc8)
    b8_hist = tf.histogram_summary("fc8_biases",b_fc8)
    y_conv=tf.nn.softmax(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)

    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-15,1.0)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    merged = tf.merge_all_summaries()
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("./tmp/deep_mnist_20160112",sess.graph_def)
    print mnist.train._num_examples
    for i in range(5000):
      batch = mnist.train.next_batch(50)
      print batch[0][0]
      if i%100 == 0:
        result = sess.run([merged])
        writer.add_summary(result[0],i)
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print "finish training"
    print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


if __name__ == "__main__":
    deep_cnn_test()
