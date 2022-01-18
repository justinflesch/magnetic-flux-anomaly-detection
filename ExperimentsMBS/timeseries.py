from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *
from rulsif import *

# Hyperparameters
SampleWidth = 10
RetroWidth = 10
Sigma = 1 # I have no idea what this should be
Alpha = 0.5 # Also no real clue here
Lambda = 0.01

def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    #ax1.plot(range(len(data)), data, c=color, alpha=0.25, label="STD's from mean")
    ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    #ax1.axhline(y=anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    #ax1.axhline(y=-anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("STDs")
    ax1.tick_params(axis='y')
    plt.show()

def quadPlot(data1, data2Tuple, vLines = []):
    total = np.transpose(data2Tuple[:,0,:])
    forward = np.transpose(data2Tuple[:,1,:])
    backward = np.transpose(data2Tuple[:,2,:])
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(data1, alpha=0.25, label="Y")
    ax2.plot(*total, alpha=0.25, label="Divergence")
    ax2.plot(*forward, alpha=0.25, label="Divergence")
    ax2.plot(*backward, alpha=0.25, label="Divergence")
    
    for x in vLines:
        ax1.axvline(x=x)
        ax2.axvline(x=x)
    
    plt.show()

def reduceData(data,window=100):
    length = data.shape[1] // window

    res = np.zeros((data.shape[0], length))

    for i in range(length):
        res[:, i] = np.mean(data[:, i*window:(i+1)*window], axis=1)
    
    return res

with TdmsFile.open("later.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    data = measurements.channels()[500:600]
    #data = measurements.channels()[500:501]
    
    data = np.nan_to_num(data)
    
    data = reduceData(data)
    
    X1 = np.array([[1,2],[3,4]])
    X2 = np.array([[1,2],[3,4]])
    Deltas = badSpecialMatrixSubtract(X1, X2)
    Deltas2 = vectorizedMatrixSubtract(X1, X2)
    print(Deltas)
    print(Deltas2)
    
    plotData(data[0])
    
    print(data.shape)
    print("Begin swapping")
    print(np.swapaxes(data,0,1).shape)
    results = TimeSeriesDissimilarity(np.swapaxes(data,0,1), SampleWidth, RetroWidth, Sigma, Alpha, Lambda)
    
    quadPlot(data[0], results)
    #plotData(results)