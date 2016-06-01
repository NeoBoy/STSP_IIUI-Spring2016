%% Audio Source Speration using Convolutional Neural Network %%

%References:
% 1. Probabilistic Binary-Mask Cocktail-Party Source Separation in a 
%    Convolutional Deep Neural Network (http://arxiv.org/abs/1503.06962)


% Copyright (c) 2016, Engr Sharjeel Abid Butt, PhD Scholar, Department of Electronic Engineering, 
% Faculty of Engineering and Technology, International Islamic University, Islamabad, Pakistan.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.

%% Data Initialization

clear
close all
clc

fs = 4000;

trainDuration = 120; % Audio Part for training in seconds
testDuration  = 10; % Audio Part for testing in seconds

FemaleSpeechFile = 'fsew0_001-043.wav';
MaleSpeechFile   = 'msak0_001-043.wav';

% Female Speech Conversion
[F_speech, FSf] = audioread(FemaleSpeechFile);
F_speechTrain   = F_speech(1:trainDuration * FSf);
f               = decimate(F_speechTrain, FSf / fs);

% sound(f, fs)
% clear sound

% Male Speech Conversion
[M_speech, FSm] = audioread(MaleSpeechFile);
M_speechTrain   = M_speech(1:trainDuration * FSm);
m               = decimate(M_speechTrain, FSm / fs);

% sound(m, fs)
% clear sound

% Mixture Creation
mix = m + f;

% sound(mix, fs)
% clear sound

% Spectrogram Creation
windowsize = 128;
window     = hanning(windowsize);
noverlap   = windowsize - 1;
nfft       = windowsize;

female_spec = spectrogram(f, window, noverlap, nfft, fs); 
male_spec   = spectrogram(m, window, noverlap, nfft, fs); 
mix_spec    = spectrogram(mix, window, noverlap, nfft, fs);

% Magnitude-only Spectrogram
fSpec_mag   = abs(female_spec);
mSpec_mag   = abs(male_spec);
mixSpec_mag = abs(mix_spec);

%% Training Convolutional Neural Network

windowSize      = 20;
overlapInterval = 10;


% Binary Mask (LABELs)
BinMask = mSpec_mag > fSpec_mag;

y = specWindowOperation(BinMask, windowSize, overlapInterval);

% Data Examples (Input Data)
X = specWindowOperation(mixSpec_mag, windowSize, overlapInterval);
