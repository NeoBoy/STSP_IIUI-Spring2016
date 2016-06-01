# -*- coding: utf-8 -*-
"""
The goal of this file is to implement window operation on the spectrogram

@author: Sharjeel Abid Butt
"""
import numpy as np

def specWindowOperation(spect, windowSize = 20, overlapInterval = 10, winPrint = False):
    start = 0
    stop  = windowSize
    total = np.shape(spect)[1]
    data  = [] 
    
    while(1):
        if(stop > total):            
            return np.asarray(data)
        if winPrint:
            print (start + 1, stop)
        win = spect[:, start:stop]
        win = np.reshape(win, (np.size(win)))
        data.append(win)
        start = start + overlapInterval
        stop  = start + windowSize
#
#spect  = np.random.random(size=(65,2000))
#result = specWindowOperation(spect, windowSize = 20, overlap = 10, winPrint = True)