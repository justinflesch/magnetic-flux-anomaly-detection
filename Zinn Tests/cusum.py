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
import math

def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    
    ax1.set_xlabel("Sample")
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
    ax1.plot(range(len(data1)), data1, alpha=1, label="Sensor Data")
    
    ax2.plot(range(len(data2)), data2, alpha=1, label="Sensor Data")
    
    ax2.set_xlabel("Sample")
    ax2.tick_params(axis='y')

    ax3.plot(range(len(data3)), data3, alpha=1, label="Sensor Data")
    
    ax3.set_xlabel("Sample")
    ax3.tick_params(axis='y')

    plt.show()

def quadPlot(data1, data2, data3, data4, color = None, marker = 'o'):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=1, label="Sensor Data")
    
    ax2.plot(range(len(data2)), data2, alpha=1, label="Sensor Data")
    ax2.tick_params(axis='y')

    ax3.plot(range(len(data3)), data3, alpha=1, label="Sensor Data")
    ax3.tick_params(axis='y')

    ax4.plot(range(len(data4)), data4, alpha=1, label="Sensor Data")
    ax4.tick_params(axis='y')

    plt.show()

def triplePlotCusum(data1, data2, data3, hPos, hNeg, color = None, marker = 'o'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=1, label="Sensor Data")
    
    ax2.plot(range(len(data2)), data2, alpha=1, label="Sensor Data")
    ax2.plot(range(len(data2)), hPos, alpha=0.5, label="Sensor Data")
    
    ax2.set_xlabel("Sample")
    ax2.tick_params(axis='y')

    ax3.plot(range(len(data3)), data3, alpha=1, label="Sensor Data")
    ax3.plot(range(len(data2)), hNeg, alpha=0.5, label="Sensor Data")
    
    ax3.set_xlabel("Sample")
    ax3.tick_params(axis='y')

    plt.show()

def autoRegressMod(data, windowSize = 3):
    moddedData = np.zeros((data.shape[0], data.shape[1]-windowSize))

    # For each sensor
    for channel in range(0, data.shape[0]):
        slidingWindow = np.lib.stride_tricks.sliding_window_view(data[channel][:-1], windowSize)

        y = data[channel][windowSize:]

        reg = LinearRegression().fit(slidingWindow, y)

        expectedVals = reg.predict(slidingWindow)

        moddedData[channel] = y - expectedVals

    return moddedData

def linearMod(data, windowWidth = 500):
    channelCount = data.shape[0]
    windowCount = (data.shape[1]) // windowWidth
    moddedData = np.zeros(data.shape)
    
    for i in range(0, windowCount+1):
        startPoint = windowWidth * i
        endPoint = min(windowWidth * (i+1), data.shape[1])

        dataset = data[:, startPoint:endPoint]
        
        X = np.arange(0, dataset.shape[1])
        fit = np.polyfit(X, dataset.T, 1).T
        m = fit[:,0][:, None]
        b = fit[:,1][:, None]
        modMatrix = (X[:, None] @ m.T).T + b
        
        moddedData[:, startPoint:endPoint] = dataset - modMatrix
    
    return moddedData

# Running average/std
def cusumRunning(data, hMult, kMult = 1):
    qsum = np.zeros(data.shape[1])
    
    # Going through each channel
    for channel in range(0, data.shape[0]):
        # Arrays to be populated with cusum
        highsum = np.zeros(data.shape[1])
        lowsum = np.zeros(data.shape[1])

        hPos = np.zeros(data.shape[1])
        hNeg = np.zeros(data.shape[1])

        # Running mean and std deviation of channel
        total = data[channel][0]
        squaresTotal = (data[channel][0])**2
        mean = data[channel][0]
        std = 0

        # Populate cusum array
        for i in range(1, data.shape[1]):
            # Current measurement, x
            x = data[channel][i]

            # Update running mean and standard deviation
            total = total + x
            squaresTotal = squaresTotal + (x ** 2)

            mean = total/(i+1)
            var = ((1 / (i+1)) * squaresTotal) - (mean ** 2)
            std = math.sqrt(abs(var))

            # Size of shift to be detected, k
            k = kMult * std

            # Control limit, h
            h = hMult * std

            hPos[i] = h
            hNeg[i] = (-1) * h

            highsum[i] = max(0, highsum[i-1] + x - mean - k)
            lowsum[i] = min(0, lowsum[i-1] + x - mean + k)

            # If measurement i is "anomalous," increase qsum[i] by one
            if highsum[i] > h:
                qsum[i] = qsum[i] + 1
                # highsum[i] = 0

            if lowsum[i] < (-1 * h):
                qsum[i] = qsum[i] + 1
                # lowsum[i] = 0

        # triplePlotCusum(data[channel], highsum, lowsum, hPos, hNeg)

    return qsum

def cusumChannel(dataset, qsum, hMult, kMult):
    # Arrays to be populated with cusum
    highsum = np.zeros(dataset.shape[0])
    lowsum = np.zeros(dataset.shape[0])

    # Mean and std deviation of channel
    mean = np.mean(dataset)
    std = np.std(dataset)

    # Size of shift to be detected, k
    k = kMult * std

    # Control limit, h
    h = hMult * std

    # Populate cusum arrays
    for i in range(1, dataset.shape[0]):
        highsum[i] = max(0, highsum[i-1] + dataset[i] - mean - k)
        lowsum[i] = min(0, lowsum[i-1] + dataset[i] - mean + k)

    # If measurement i is "anomalous," increase qsum[i] by one
    for i in range(1, dataset.shape[0]):
        if highsum[i] > h:
            qsum[i] = qsum[i] + 1
            # highsum[i] = 0

        if lowsum[i] < (-1 * h):
            qsum[i] = qsum[i] + 1
            # lowsum[i] = 0

    hPos = np.full(dataset.shape[0], h)
    hNeg = np.full(dataset.shape[0], ((-1) * h))
    # triplePlotCusum(dataset, highsum, lowsum, hPos, hNeg)

def cusum(data, hMult, kMult = 1):
    qsum = np.zeros(data.shape[1])
    
    # Going through each channel
    for channel in range(0, data.shape[0]):
        # Run cusum for channel
        cusumChannel(data[channel], qsum, hMult, kMult)

    qsumEnd = np.zeros(data.shape[1])

    return qsum

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    #data = measurements.channels()[500:510]
    data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)

    # Testing
    # data = data[:,1200:]

    # Normalize data
    # data = (data - data.mean())/(data.std())

    # Do autoregression to remove linear trend
    # data = autoRegressMod(data)
    
    # Run cusum
    hVal = 10

    cusumed = cusum(data, hVal)

    runningcusumed = cusumRunning(data, hVal)

    triplePlot(measurements.channels()[0], cusumed, runningcusumed)

    # triplePlot(measurements.channels()[0][1200:], data[0], cusumed)