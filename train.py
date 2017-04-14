import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000, help='number of training iterations')
parser.add_argument('--sample', type=int, default=100, help='rate current cost is evaluated')
parser.add_argument('--batch', type=int, default=50, help='number of examples in each training epoch')
parser.add_argument('--plot', dest='plot', action='store_true', help='plot cost history graph')
parser.set_defaults(plot=False)
args = parser.parse_args()

training_epochs = args.epochs
sample_size = args.sample
plot_cost = args.plot
batch_size = args.batch

flat_dimension = 10000
model_dir = 'model'
log_dir = 'log'
folds = [0, 0, 0, 0]

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sess = tf.InteractiveSession()

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

def slice(index, features, labels, size):
    """
    Slices `size` items from `features` and `labels` from the `index` backwards.
    For example, index=10, size=2, returns features[8:10], labels[8:10]

    Args:`
        index (int): The current index
        features (array): The features
        labels (array): The labels
        size (int): The size of slice to return

    Returns:
        array, array: A tuple containing arrays of (features, labels)
    """
    x = max((index - size), 0)
    y = max(index, 0)
    batch_x = features[x:y]
    batch_y = labels[x:y]
    return batch_x, batch_y

def get_next_training_batch(size):
  """
    Returns a batch of training examples from fold1, fold2, or fold3. Batch
    is of size specified and on a moving window through each fold. When one
    fold is exhausted it will move onto the next fold. When all are exhausted
    it will restart with the first fold.

    Args:

    Returns:
        array, array: A tuple containing arrays of (features, labels)
    """
  if ((folds[0] >= 0) and (folds[1] >= 0) and (folds[2] >= 0)):
    folds[0] = len(fold2_features)
    folds[1] = len(fold2_features)
    folds[2] = len(fold3_features)

  if (folds[0] > 0):
    batch_x, batch_y = slice(folds[0], fold1_features, fold1_labels, size)
    folds[0] = max(folds[0] - size, 0)
  elif (folds[1] > 0):
    batch_x, batch_y = slice(folds[1], fold2_features, fold1_labels, size)
    folds[1] = max(folds[1] - size, 0)
  elif (folds[2] > 0):
    batch_x, batch_y = slice(folds[2], fold1_features, fold1_labels, size)
    folds[2] = max(folds[2] - size, 0)

  return batch_x, batch_y

def get_next_validation_batch(size):
  """
    Returns a batch of validation examples from fold4. Batch is of size specified 
    and on a moving window through fold4. When the fold is exhausted it will restart.

    Args:

    Returns:
        array, array: A tuple containing arrays of (features, labels)
    """
  if (folds[3] <= 0):
    folds[3] = len(fold4_features)

  batch_x, batch_y = slice(folds[3], fold4_features, fold4_labels, size)
  folds[3] = max(folds[3] - size, 0)

  return batch_x, batch_y

# load data
print("loading training data")
fold1_features = np.load(file='data/fold1_features.npy')
fold1_labels = np.load(file='data/fold1_labels.npy')
folds[0] = len(fold1_features)

fold2_features = np.load(file='data/fold2_features.npy')
fold2_labels = np.load(file='data/fold2_labels.npy')
folds[1] = len(fold2_features)

fold3_features = np.load(file='data/fold3_features.npy')
fold3_labels = np.load(file='data/fold3_labels.npy')
folds[2] = len(fold3_features)

num_training_examples = folds[0] + folds[1] + folds[2]
print("training example size: " + str(num_training_examples))

print("loading validation data")
fold4_features = np.load(file='data/fold4_features.npy')
fold4_labels = np.load(file='data/fold4_labels.npy')
folds[3] = len(fold4_features)

num_validation_examples = folds[3]
print("validation example size: " + str(num_validation_examples))

num_labels = len(fold1_labels[0])
print("found " + str(num_labels) + " unique labels in dataset")

# graph nodes
x = tf.placeholder(tf.float32, shape=[None, flat_dimension])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

W = tf.Variable(tf.zeros([flat_dimension, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

y = tf.matmul(x,W) + b
 
# first convolutional layer - patch size 5x5, input channel 1, output channel 32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# apply layer
x_image = tf.reshape(x, [-1, 100, 100, 1])

# convolve, add bias, apply ReLU, max pool - reduces image to 50x50
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer - patch size 5x5, input channel 32, output channel 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# reduced to 25x25

# densely connected layer
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
with tf.name_scope('Dropout'):
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
tf.summary.scalar('dropout_keep_probability', keep_prob)

# regression layer
W_fc2 = weight_variable([1024, num_labels])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaulate model
with tf.name_scope('Cross_Entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('Accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

print("running training: epochs[" + str(training_epochs) + "] batch[" + str(batch_size) + "] sample[" + str(sample_size) + "]")
start = int(round(time.time() * 1000))
sess.run(tf.global_variables_initializer())
loss_history = np.empty(shape=[1], dtype=float)

train_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
merged = tf.summary.merge_all()

for i in range(training_epochs):
  batch_x, batch_y = get_next_training_batch(batch_size)

  if i%sample_size == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
    loss = (1 - train_accuracy)
    if (plot_cost):
      loss_history = np.append(loss_history, loss)
    print("epoch: " + str(i) + " loss: " + str(loss))
  
  summary, acc = sess.run([merged, train_step], feed_dict={x:batch_x, y_:batch_y, keep_prob: 0.5})
  train_writer.add_summary(summary, i)

# evaulate accuracy with validation set
v_accuracy = []
for count in range(0, 6):
  v_batch_x, v_batch_y = get_next_validation_batch(batch_size)
  v_accuracy.append(accuracy.eval(feed_dict={x:v_batch_x, y_:v_batch_y, keep_prob: 1.0}))
print("validation accuracy: ", np.mean(v_accuracy))

saver = tf.train.Saver()
saver.save(sess, model_dir + '/whatson')

end = int(round(time.time() * 1000))
print("completed in " + str((end - start) / 1000) + " seconds")

if (plot_cost):
    figure = plt.figure(figsize=(10, 8))
    plt.plot(loss_history)
    plt.axis([0, (training_epochs/sample_size), 0, np.max(loss_history)])
    plt.show()
