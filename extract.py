import glob
import os
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
import numpy as np
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='input file directory')
args = parser.parse_args()

audio_dir = args.dir
dest_dir = 'data'

flat_dimension=10000 # = 3.85356455 seconds
num_unique_classes=5

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

def extract_label(filename):
    labels = filename.split('/')[::-1][0].split('.')[0].split('-')[2]
    return labels

def one_hot_encode(class_num):
    n = int(class_num)
    one_hot_encode = np.zeros(num_unique_classes,)
    one_hot_encode[n] = 1
    return one_hot_encode

def extract(filename, max_dimension):
    y, sr = librosa.load(filename)
    label = filename.split('/')[::-1][0].split('.')[0].split('-')[2]
    one_hot_label = one_hot_encode(label)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
    flat_mfcc = mfcc.flatten()
    return one_hot_label, flat_mfcc[0:max_dimension]

def process_files(in_dir, max_dimension):
    labels = []
    melspecs = []
    for file in glob.glob(os.path.join(in_dir, "*.wav")):
        print("extracting feature from ", file)
        label, melspec = extract(file, max_dimension)
        labels.append(label)
        melspecs.append(melspec)
    return np.array(labels), np.array(melspecs)

def save_data(filename, labels, features):
    np.save(file="data/" + filename + "_labels", arr=labels)
    np.save(file="data/" + filename + "_features", arr=features)

start = int(round(time.time() * 1000))

print("extracting training features and labels")
training_labels, training_features = process_files(audio_dir + 'train', flat_dimension)
print("training example size: " + str(len(training_features)))
print("training feature dimension: " + str(len(training_features[0])))
save_data('training', training_labels, training_features)

print("extracting validation features and labels")
validation_labels, validation_features = process_files(audio_dir + 'valid', flat_dimension)
print("validation example size: " + str(len(validation_features)))
print("validation feature dimension: " + str(len(training_features[0])))
save_data('validation', validation_labels, validation_features)

end = int(round(time.time() * 1000))

print("completed in " + str((end - start) / 1000) + " seconds")
