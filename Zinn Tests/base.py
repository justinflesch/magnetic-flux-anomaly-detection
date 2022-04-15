from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture

def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("STDs")
    ax1.tick_params(axis='y')
    plt.show()

def doublePlot(data1, data2, color = None, marker = 'o'):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=0.25, label="Sensor Data")
    
    ax2.scatter(list(range(len(data2))), data2, c = color, cmap = 'seismic', marker=marker)  
    
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("STDs")
    ax2.tick_params(axis='y')

    plt.show()

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    data = measurements.channels()[500:510]
    
    data = np.nan_to_num(data)

    # Makes sliding window of data, takes mean of each window, substracts each data point by the mean of the sliding window it begins at,
    # then divides it by the window's stddev to normalize it

    # Sliding window
    slidingWindow = np.lib.stride_tricks.sliding_window_view(data, 50, axis=1)

    # Mean/stddev/variance of each window
    windowMean = np.mean(slidingWindow, axis=2)
    # windowStd = np.std(slidingWindow, axis=2)
    # windowVar = np.var(slidingWindow, axis=2)

    data = data[:,0:windowMean.shape[1]]
    result = data - windowMean
    # result = result/windowStd

    # (observatoin - mean) / stddev

    doublePlot(measurements.channels()[0], result[0])

    # Take transpose for sklearn
    transposeData = np.transpose(result)

    #-------------------------------------------------------------------------------------------------------------------------------------------

    # Elliptic Envelope
    elp = EllipticEnvelope()
    ret = elp.fit_predict(transposeData)
    ret = elp.score_samples(transposeData)

    doublePlot(measurements.channels()[0], ret)