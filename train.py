import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

sess = tf.InteractiveSession()

training_epochs = 2000
sample_size = 1
batch_size = 50
dimensions = 784

def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)
 
def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)
 
def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def plot_history(history):
    figure = plt.figure(figsize=(10, 8))
    plt.plot(history)
    plt.axis([0, (training_epochs/sample_size), 0, np.max(history)])
    plt.show()

# load data
print("loading training data")
training_features = np.load(file='data/training_features.npy')
training_labels = np.load(file='data/training_labels.npy')
num_training_examples = len(training_features)
print("training example size: " + str(num_training_examples))

print("loading validation data")
validation_features = np.load(file='data/validation_features.npy')
validation_labels = np.load(file='data/validation_features.npy')
num_validation_examples = len(validation_features)
print("validation example size: " + str(num_validation_examples))

num_labels = len(training_labels[0])
print("found " + str(num_labels) + " unique labels in dataset")

# graph nodes
x = tf.placeholder(tf.float32, shape=[None, dimensions])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

W = tf.Variable(tf.zeros([dimensions, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

y = tf.matmul(x,W) + b
 
# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# apply layer
x_image = tf.reshape(x, [-1,28,28,1])

# convolve, add bias, apply ReLU, max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer (regression)
W_fc2 = weight_variable([1024, num_labels])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaulate model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("running training for " + str(training_epochs) + " epochs")
start = int(round(time.time() * 1000))
sess.run(tf.global_variables_initializer())
loss_history = np.empty(shape=[1], dtype=float)

for i in range(training_epochs):
  offset = (i * batch_size) % (num_training_examples - batch_size)
  batch_x = training_features[offset:(offset + batch_size)]
  batch_y = training_labels[offset:(offset + batch_size)]

  if i%100 == 0:
    print("epoch: " + str(i))

  if i%sample_size == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
    loss = (1 - train_accuracy)
    loss_history = np.append(loss_history, loss)
  train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob: 0.5})

end = int(round(time.time() * 1000))
print("completed in " + str((end - start) / 1000) + " seconds")

plot_history(loss_history)
