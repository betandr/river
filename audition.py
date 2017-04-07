import glob
import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa


Y = tf.placeholder(tf.float32,[None, num_classes])

out_weights = weights([num_hidden, num_labels])
out_biases = bias([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)


saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "model/model.ckpt")
  print("Model restored.")
  correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))