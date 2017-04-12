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
parser.add_argument('--type', type=str, default='spec')
parser.add_argument('--title', type=str, default='Plot')
parser.add_argument('--file', action='append', required=True)
args = parser.parse_args()

files = args.file
plot_type = args.type
title = args.title

def load_sound_files(files):
    raw_sounds = []
    for fp in files:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waveform(raw_sounds):
    i = 1
    fig = plotter.figure(figsize=(25, 60))
    fig.suptitle("Waveform " + title, fontsize=12)
    for f in raw_sounds:
        plotter.subplot(2, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        i += 1
    plotter.show()

def plot_spectrogram(raw_sounds):
    i = 1
    fig = plotter.figure(figsize=(25, 60))
    fig.suptitle("Spectrogram " + title, fontsize=12)
    for sound in raw_sounds:
        plotter.subplot(2, 1, i)
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
