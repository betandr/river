# Whatson - Automated Audio Content Analysis Using Convolutional Deep Neural Networks

## Installation

```
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages ~/tensorflow
pip3 install --upgrade matplotlib
pip3 install --upgrade tensorflow
pip3 install --upgrade librosa
```

## Creating the Dataset

Whatson analyses audio files in WAVE (.wav) format with filenames like `yyyymmdd-aaaa-bb.wav` where `yyyymmdd` is a date/time, `aaaa` is an integer index, and `bb` is the zero-indexed class number for this audio sample. The only really important part is the class number but the files should be in the format `n-n-n.wav` with the last `n` the class. These should be split into a training and validation set, usually at an 80/20 ratio. This can be done by _sampling_ the files:

### Sampling the files

To sample at an 80/20 split use a stride of 5 which will move ever 5th file to the target directory. This will create an even distribution
of files for the 20% split—the remainder is 80%.
```
python3 sample_files.py --src=/path/to/wav/input/files/ --stride=2000 --dest=/path/to/wav/output/files/
```

## Activating Tensorflow

```
source ~/tensorflow/bin/activate
```

## Running
### 0) Plotting
```
python3 plot.py --type{wave|spec} --file=/path/to/file1.wav --file=/path/to/file2.wav
```
![Waveform Plot of R&B Track](https://github.com/betandr/whatson/blob/master/images/randb_wave.png)
_Waveform Plot of R&B Track_

![Spectrogram Plot of R&B Track](https://github.com/betandr/whatson/blob/master/images/randb_spec.png)
_Spectrogram Plot of R&B Track_

![Waveform Plot of Speech](https://github.com/betandr/whatson/blob/master/images/speech_wave.png)
_Waveform Plot of Speech_

![Spectrogram Plot of Speech](https://github.com/betandr/whatson/blob/master/images/speech_spec.png)
_Spectrogram Plot of Speech_

### 1) Audio Analysis
Extract features found in `train` and `valid` subdirectories. Creates training and validation datasets in the `data` directory
```
python3 extract.py --dir=/path/to/audio
```

### 2) Training neural network
Use training and validation datasets from the `data` directory to train a convolutional neural network. 
Saves model in `model` directory. 
* `epochs` - number of training iterations
* `batch` - number of examples to supply in each training epoch 
* `sample` - how often the current cost is returned to the console/stored for plotting (if `--plot` supplied)
* `plot` - supplied if a cost history graph is required or omitted if not.

```
python3 train.py --epochs=2000 --batch=50 --sample_size=10 --plot
```
Logs during training will be created in the `log` subdirectory, these can be visualised with [TensorBoard](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md)

## Evaluating

Run TensorBoard:
```
tensorboard --logdir=log/
```

View results at `http://localhost:6006/`

![Accuracy Plot](https://github.com/betandr/whatson/blob/master/images/accuracy.png)

## Deactivating Tensorflow

```
deactivate
```
