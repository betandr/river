import numpy as np
import tensorflow as tf
import time
import argparse
import os
import librosa
import operator

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='input wave file')
args = parser.parse_args()

file = args.file

model_dir = 'model'

flat_dimension = 10000
num_labels=5

def decode_prediction(pred):
  index, value = max(enumerate(pred), key=operator.itemgetter(1))
  return index, value

def extract_from_file(filename, max_dimension):
  melspecs = []
  y, sr = librosa.load(filename)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
  flat_mfcc = mfcc.flatten()
  melspec = flat_mfcc[0:max_dimension]
  melspecs.append(melspec)
  return np.array(melspecs)

# graph nodes
x = tf.placeholder(tf.float32, shape=[None, flat_dimension])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
W = tf.Variable(tf.truncated_normal([flat_dimension, num_labels], stddev=1./22.))
b = tf.Variable(tf.truncated_normal([num_labels], stddev=1./22.))
y = tf.matmul(x,W) + b
keep_prob = tf.placeholder(tf.float32)

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, model_dir + "/whatson")

  feature = extract_from_file(file, flat_dimension)
  
  feed_dict = {x: feature, keep_prob:1.0}
  classification = y.eval(feed_dict, sess)

  prediction, confidence = decode_prediction(classification[0].tolist())

  print("prediction: class " + str(prediction) + " with " + str(confidence) + " confidence")
