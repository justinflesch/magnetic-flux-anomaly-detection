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
    ax1.plot(range(len(data1)), data1, alpha=1, label="Sensor Data")
    
    ax2.plot(range(len(data2)), data2, alpha=1, label="Sensor Data")
    
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("STDs")
    ax2.tick_params(axis='y')

    ax3.plot(range(len(data3)), data3, alpha=1, label="Sensor Data")
    
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("STDs")
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

def fivePlot(data1, data2, data3, data4, data5, color = None, marker = 'o'):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=1, label="Sensor Data")
    
    ax2.plot(range(len(data2)), data2, alpha=1, label="Sensor Data")
    ax2.tick_params(axis='y')

    ax3.plot(range(len(data3)), data3, alpha=1, label="Sensor Data")
    ax3.tick_params(axis='y')

    ax4.plot(range(len(data4)), data4, alpha=1, label="Sensor Data")
    ax4.tick_params(axis='y')

    ax5.plot(range(len(data5)), data5, alpha=1, label="Sensor Data")
    ax5.tick_params(axis='y')

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

with TdmsFile.open("fullmelt.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)
    
    autoRegress = autoRegressMod(data, 3)
    # autoRegress1 = autoRegressMod(data, 1)
    # autoRegress2 = autoRegressMod(data, 5)
    # autoRegress3 = autoRegressMod(data, 10)
    # autoRegress4 = autoRegressMod(data, 100)

    doublePlot(measurements.channels()[0], autoRegress[0])

    for i in range(0, data.shape[1]):
        # fivePlot(data[i], autoRegress1[i], autoRegress2[i], autoRegress3[i], autoRegress4[i],)
        doublePlot(data[i], autoRegress[i])