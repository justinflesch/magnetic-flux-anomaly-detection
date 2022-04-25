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
from sklearn.linear_model import LinearRegression

def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("STDs")
    ax1.tick_params(axis='y')
    plt.show()

def doublePlot(data1, data2, color = None, marker = 'o'):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(len(data1)), data1)
    
    ax2.plot(range(len(data2)), data2)  
    
    ax1.set_xlabel("Sample")
    ax2.set_xlabel("Sample")
    ax2.tick_params(axis='y')

    plt.show()

def triplePlot(data1, data2, data3, color = None, marker = 'o'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=0.25, label="Sensor Data")
    
    ax2.scatter(list(range(len(data2))), data2, c = color, cmap = 'seismic', marker=marker)  
    
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("STDs")
    ax2.tick_params(axis='y')

    ax3.scatter(list(range(len(data3))), data3, c = color, cmap = 'seismic', marker=marker)  
    
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("STDs")
    ax3.tick_params(axis='y')

    plt.show()

# Returns a 2d array (approximately) the size of the data passed in
# Does an autoregression on the data using the last windowSize values to get expectedVals, then returns the difference data - expectedVals
# Therefore, just leaves the residuals
def autoRegressMod(data, windowSize = 3):
    moddedData = np.zeros((data.shape[0], data.shape[1]-windowSize))

    # For each sensor
    for channel in range(0, data.shape[0]):
        # Creates 2d array with windowSize columns. For example, if the original dataset was 1, 2, 3, 4 ... slidingWindow would be 
        # [1 2]
        # |2 3|
        # |3 4|
        # [...]
        slidingWindow = np.lib.stride_tricks.sliding_window_view(data[channel][:-1], windowSize)

        # y is a 1d array of the expected values (the values for each time tick)
        y = data[channel][windowSize:]

        # Fits a linear regression using the values from slidingWindow to predict the values in y
        reg = LinearRegression().fit(slidingWindow, y)

        expectedVals = reg.predict(slidingWindow)

        # Once the expected values are found via the autoregression, take the difference between
        # the original values and expectedVals to get the residual. Fill moddedData with the channel's residuals
        moddedData[channel] = y - expectedVals

    return moddedData

# Does a linear regression on each window of size windowWidth to get expectedVals
# Then returns data - expectedVals to leave residuals
def linearMod(data, windowWidth = 500):
    windowCount = (data.shape[1]) // windowWidth
    moddedData = np.zeros(data.shape)
    
    # For each window
    for i in range(0, windowCount+1):
        # Get start and end index of the current window
        startPoint = windowWidth * i
        endPoint = min(windowWidth * (i+1), data.shape[1])

        dataset = data[:, startPoint:endPoint]
        
        # Uses really weird syntax + linear algebra stuff to make an array modMatrix
        # filled with the expected values found via polyfit
        X = np.arange(0, dataset.shape[1])
        fit = np.polyfit(X, dataset.T, 1).T
        m = fit[:,0][:, None]
        b = fit[:,1][:, None]
        modMatrix = (X[:, None] @ m.T).T + b
        
        # Takes difference between dataset and modMatrix to get residuals
        moddedData[:, startPoint:endPoint] = dataset - modMatrix
    
    return moddedData

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    #data = measurements.channels()[500:510]
    data = measurements.channels()[500:510]
    
    data = np.nan_to_num(data)
    
    autoRegress = autoRegressMod(data, 5)
    linear = linearMod(data)

    ##########################################################

    # # Sliding window
    # slidingWindow = np.lib.stride_tricks.sliding_window_view(data, 50, axis=1)

    # # Mean/stddev/variance of each window
    # windowMean = np.mean(slidingWindow, axis=2)
    # # windowStd = np.std(slidingWindow, axis=2)
    # # windowVar = np.var(slidingWindow, axis=2)

    # autoRegress = autoRegress[:,0:windowMean.shape[1]]
    # result = autoRegress - windowMean

    # normed = (autoRegress - autoRegress.mean())/(autoRegress.std())

    # doublePlot(autoRegress[0], result[0])

    ##########################################################
    
    transposeData = autoRegress.T
    
    elp = EllipticEnvelope(contamination = 0.03)
    elp.fit(transposeData)
    ret = elp.score_samples(transposeData)
    doublePlot(measurements.channels()[0], ret)
    