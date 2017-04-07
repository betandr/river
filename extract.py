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
parser.add_argument('--dir', type=str)
args = parser.parse_args()

audio_dir = args.dir
dest_dir = 'data'

dimension=28
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

def extract(filename, dimension):
    flat_dimension = dimension * dimension
    sound_clip, s = librosa.load(filename)
    label = filename.split('/')[::-1][0].split('.')[0].split('-')[2]
    one_hot_label = one_hot_encode(label)
    cropped = sound_clip[int(0):int(flat_dimension/2)]
    melspec = librosa.feature.melspectrogram(cropped, n_mels=flat_dimension)
    logspec = librosa.logamplitude(melspec.flatten())
    return one_hot_label, logspec

def process_files(in_dir, dimension):
    labels = []
    logspecs = []
    for file in glob.glob(os.path.join(in_dir, "*.wav")):
        print("extracting feature from ", file)
        label, logspec = extract(file, dimension)
        labels.append(label)
        logspecs.append(logspec)
    return np.array(labels), np.array(logspecs)

def save_data(filename, labels, features):
    np.save(file="data/" + filename + "_labels", arr=labels)
    np.save(file="data/" + filename + "_features", arr=features)

start = int(round(time.time() * 1000))

print("extracting training features and labels")
training_labels, training_features = process_files(audio_dir + 'train', dimension)
print("training example size: " + str(len(training_features)))
save_data('training', training_labels, training_features)

print("extracting validation features and labels")
validation_labels, validation_features = process_files(audio_dir + 'valid', dimension)
print("training validation size: " + str(len(validation_features)))
save_data('validation', validation_labels, validation_features)

end = int(round(time.time() * 1000))

print("completed in " + str((end - start) / 1000) + " seconds")
