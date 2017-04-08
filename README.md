# Whatson - Automated Audio Content Analysis Using Convolutional Deep Neural Networks

## Installation

```
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages ~/tensorflow
pip3 install --upgrade matplotlib
pip3 install --upgrade tensorflow
pip3 install --upgrade librosa
```

## Activating Tensorflow

```
source ~/tensorflow/bin/activate
```

## Running
### 0) Plotting
```
python3 plot.py --type{wave|spec} --file=path/file1.wav --file=path/file2.wav
```
![Waveform Plot](https://github.com/betandr/whatson/blob/master/images/waveforms.png)
_Waveform Plot_

![Waveform Plot](https://github.com/betandr/whatson/blob/master/images/spectrograms.png)
_Spectrogram Plot_

### 1) Audio Analysis
Extract features found in `train` and `valid` subdirectories. Creates 
training and validation datasets in the `data` directory
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

![Cost History Plot](https://github.com/betandr/whatson/blob/master/images/cost_history.png)

## Deactivating Tensorflow

```
deactivate
```
