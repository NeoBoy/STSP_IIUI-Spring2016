# -*- coding: utf-8 -*-
"""
%% Audio Source Speration using Convolutional Neural Network %%


Desinging a Convolutional Neural Network for Audio Source Seperation using
a binary Mask
 

@author: Sharjeel Abid Butt

@References

1. Probabilistic Binary-Mask Cocktail-Party Source Separation in a 
   Convolutional Deep Neural Network (http://arxiv.org/abs/1503.06962)
   
"""

## Clear function
__saved_context__ = {}

def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

clear = restoreContext
saveContext()


clear()  # Calling clear for clearing workspace


## Importing Libraries

import numpy as np
import stft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd

import sounddevice as sd
import mir_eval #BSS-EVAL equivalent

from scipy.signal import decimate, hanning
from sklearn.preprocessing import scale

from specWindowOperation import specWindowOperation


## Data Initialization
fs = 4000

trainDuration = 120 # Audio Part for training in seconds
testDuration  = 10  # Audio Part for testing in seconds

FemaleSpeechFile = 'fsew0_001-043.wav'
MaleSpeechFile   = 'msak0_001-043.wav'

# Female Speech Conversion
FSf, F_speech = wav.read(FemaleSpeechFile)

# Normalize the audio data
F_speech      = F_speech / 2.0 ** 15
F_speechTrain = F_speech[0:trainDuration * FSf]
f             = decimate(F_speechTrain, FSf / fs)
#f = scale(f)
#f = f.astype(np.int16)

# sd.play(f, fs)
# sd.stop()

# Male Speech Conversion
FSm, M_speech = wav.read(MaleSpeechFile)

# Normalize the audio data
M_speech      = M_speech / 2.0 ** 15
M_speechTrain = M_speech[0:trainDuration * FSm]
m             = decimate(M_speechTrain, FSm / fs)
#m = scale(m)
#m = m.astype(np.int16)

# sd.play(m, fs)
# sd.stop()

# Mixture Creation
mix = m + f
#mix = mix.astype(np.int16)

# sd.play(mix, fs)
# sd.stop()

# Spectrogram Creation
windowSize = 128
noverlap   = windowSize - 1
nfft       = windowSize



female_spec = stft.spectrogram(f,  framelength = windowSize, overlap = noverlap, 
                               window = hanning) 

male_spec   = stft.spectrogram(m,  framelength = windowSize, overlap = noverlap, 
                               window = hanning) 

mix_spec    = stft.spectrogram(mix,  framelength = windowSize, overlap = noverlap, 
                               window = hanning) 


# Magnitude-only Spectrogram
fSpec_mag   = np.abs(female_spec)
mSpec_mag   = np.abs(male_spec)
mixSpec_mag = np.abs(mix_spec)


## Training Convolutional Neural Network

windowSize      = 20
overlapInterval = 10

# Binary Mask (LABELs)
BinMask = mSpec_mag > fSpec_mag

y = specWindowOperation(BinMask, windowSize = windowSize, 
                        overlapInterval = overlapInterval)

# Data Examples (Input Data)
X = specWindowOperation(mixSpec_mag, windowSize = windowSize, 
                        overlapInterval = overlapInterval)
