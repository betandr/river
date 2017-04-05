import glob
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plotter
import numpy as np
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import tensorflow as tf
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--file', action='append')
args = parser.parse_args()

files = args.file
plot_type = args.type

def load_sound_files(files):
    raw_sounds = []
    for fp in files:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waveform(raw_sounds):
    i = 1
    fig = plotter.figure(figsize=(25, 60))
    fig.suptitle('Audio Waveforms', fontsize=12)
    for f in raw_sounds:
        plotter.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        i += 1
    plotter.show()

def plot_spectrogram(raw_sounds):
    i = 1
    fig = plotter.figure(figsize=(25, 60))
    fig.suptitle('Audio Spectrograms', fontsize=12)
    for sound in raw_sounds:
        plotter.subplot(10, 1, i)
        specgram(np.array(sound), Fs=22050)
        i += 1
    plotter.show()

raw = load_sound_files(files)

if(plot_type == "wave"):
    plot_waveform(raw)
elif(plot_type == "spec"):
    plot_spectrogram(raw)
else:
    print("Unknown plot type: ", plot_type)
