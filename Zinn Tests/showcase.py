from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *
import math

from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression

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

# Control chart anomaly detection
# For each channel, gets an upper control limit (UCL) and lower control limit (LCL) based on running mean and stdMult * standard deviation
# If the reading for that time value are above or below the UCL or LCL, that channel is marked anomalous for the given time value
# If a percentage of the channels are anomalous for a time value (based on sndAlarm), then the time value is marked anomalous
# Note: This currently returns ctrlAvg, where ctrlAvg[i] = fraction of channels that found time i to be anomalous
# However, changing the return to anomalies will return an array of 1's and 0's where anomalies[i] = 1 indicates time i is anomalous
def ctrlChart(data, stdMult = 2, sndAlarm = 0.25):
    ctrl = np.zeros(data.shape[1])

    # Going through each channel for 1st level alarm
    for channel in range(0, data.shape[0]):
        # Running mean and std deviation of channel
        total = data[channel][0]
        squaresTotal = (data[channel][0])**2
        mean = data[channel][0]
        std = 0

        # Going through each reading for the channel
        for i in range(1, data.shape[1]):
            # Current measurement, x
            x = data[channel][i]

            # Update running mean/std
            total = total + x
            squaresTotal = squaresTotal + (x ** 2)

            mean = total/(i+1)
            var = ((1 / (i+1)) * squaresTotal) - (mean ** 2)
            std = math.sqrt(abs(var))

             # Compute upper control limit (ucl) and lower control limit (lcl)
            lcl = mean - (stdMult * std)
            ucl = mean + (stdMult * std)

            # If it's not inside the boundaries indicated by lcl and ucl, mark ctrl[i] with +1
            if((x > ucl) or (x < lcl)):
                ctrl[i] = ctrl[i] + 1
    
    # Second level alarm

    # ctrlAvg is the fraction of channels that found time x to be anomalous. 
    # For example ctrlAvg[i] = 0.75 means 75% of the channels found time i to be anomalous on the 1st level alarm
    ctrlAvg = np.zeros(data.shape[1])
    for i in range(0, data.shape[1]):
        ctrlAvg[i] = ctrl[i] / data.shape[0]

    # If ctrlAvg[i] exceeds sndAlarm, then anomalies[i] = 1, and i is considered anomalous. Otherwise, anomalies[i] = 0.
    anomalies = np.zeros(data.shape[1])
    for i in range(0, data.shape[1]):
        if(ctrlAvg[i] > sndAlarm):
            anomalies[i] = 1

    # Currently returns ctrlAvg for more detailed results.
    # Can change this return to be anomalies to have a more binary yes/no answer. TBD
    return ctrlAvg

# Cusum anomaly detection with running mean / standard deviation
# For each channel, keeps a highsum and a lowsum.
# The highsum is defined recursively as highsum[i] = max(0, highsum[i-1] + data[i] - mean - kMult*std)
# The lowsum is defined recursively as lowsum[i] = min(0, lowsum[i-1] + data[i] - mean + kMult*std)
# If highsum[i] > hMult*std or lowsum[i] < -(hMult*std), then the channel indicates time i to be anomalous, and adds one to qsum
# Returns qsum, where qsum[i] = # of channels that found time i to be anomalous
def cusumRunning(data, hMult, kMult = 1):
    qsum = np.zeros(data.shape[1])
    
    # Going through each channel
    for channel in range(0, data.shape[0]):
        # Arrays to be populated with highsum / lowsum
        highsum = np.zeros(data.shape[1])
        lowsum = np.zeros(data.shape[1])

        # These two arrays are just used for debugging / plotting
        # hPos = np.zeros(data.shape[1])
        # hNeg = np.zeros(data.shape[1])

        # Initializing running mean and std deviation of channel
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

            # Debugging stuff
            # hPos[i] = h
            # hNeg[i] = (-1) * h

            # Populate highsum[i] and lowsum[i]
            highsum[i] = max(0, highsum[i-1] + x - mean - k)
            lowsum[i] = min(0, lowsum[i-1] + x - mean + k)

            # If measurement i is "anomalous," increase qsum[i] by one
            if highsum[i] > h:
                qsum[i] = qsum[i] + 1
                # highsum[i] = 0

            if lowsum[i] < (-1 * h):
                qsum[i] = qsum[i] + 1
                # lowsum[i] = 0

        # More debugging stuff
        # triplePlotCusum(data[channel], highsum, lowsum, hPos, hNeg)

    return qsum

# Used for constant cusum
# Runs cusum for a singular channel with constant mean / standard deviation. Read cusumRunning() or cusum() for info about cusum
# Populates qsum parameter
def cusumChannel(dataset, qsum, hMult, kMult):
    # Arrays to be populated with highsum / lowsum
    highsum = np.zeros(dataset.shape[0])
    lowsum = np.zeros(dataset.shape[0])

    # Initializing mean and std deviation of channel
    mean = np.mean(dataset)
    std = np.std(dataset)

    # Size of shift to be detected, k
    k = kMult * std

    # Control limit, h
    h = hMult * std

    # Populate highsum / lowsum arrays
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

    # Debugging stuff
    # hPos = np.full(dataset.shape[0], h)
    # hNeg = np.full(dataset.shape[0], ((-1) * h))
    # triplePlotCusum(dataset, highsum, lowsum, hPos, hNeg)

# Cusum anomaly detection with NO RUNNING MEAN / STD
# Use cusumRunning for running mean / std
# For each channel, keeps a highsum and a lowsum.
# The highsum is defined recursively as highsum[i] = max(0, highsum[i-1] + data[i] - mean - kMult*std)
# The lowsum is defined recursively as lowsum[i] = min(0, lowsum[i-1] + data[i] - mean + kMult*std)
# If highsum[i] > hMult*std or lowsum[i] < -(hMult*std), then the channel indicates time i to be anomalous, and adds one to qsum
# Returns qsum, where qsum[i] = # of channels that found time i to be anomalous
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
    
    data = measurements.channels()[500:600]
    # data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)

    # Autoregression
    autoRegress = autoRegressMod(data, 5)

    # Sliding window linear regression
    linear = linearMod(data)
    
    # Elliptic Envelope
    ellData = data.T
    elp = EllipticEnvelope()
    elp.fit_predict(ellData)
    ret = elp.score_samples(ellData)

    # Autoregression + Elliptic Envelope
    autoRegressTranspose = autoRegress.T
    elpAutoRegress = EllipticEnvelope(contamination = 0.03)
    elpAutoRegress.fit_predict(autoRegressTranspose)
    retElpAutoRegress = elpAutoRegress.score_samples(autoRegressTranspose)

    # Linear Window + Elliptic Envelope
    linearTranspose = linear.T
    linRegress = EllipticEnvelope(contamination = 0.03)
    linRegress.fit_predict(linearTranspose)
    retLinRegress = linRegress.score_samples(linearTranspose)

    # Sliding window mean difference + Elliptic Envelope
    slidingWindow = np.lib.stride_tricks.sliding_window_view(data, 50, axis=1)
    windowMean = np.mean(slidingWindow, axis=2)
    croppedData = data[:,0:windowMean.shape[1]]
    slidingDiff = croppedData - windowMean
    slidingDiffTranspose = np.transpose(slidingDiff)

    slidingElp = EllipticEnvelope()
    slidingRet = slidingElp.fit_predict(slidingDiffTranspose)
    slidingRet = slidingElp.score_samples(slidingDiffTranspose)

    # Control chart
    ctrlCharted = ctrlChart(data, 3.5, 0.35)

    # Cusum - constant mean / std
    hVal = 25
    cusumed = cusum(autoRegress, hVal)

    # Cusum - running mean / std
    runningcusumed = cusumRunning(autoRegress, hVal)

    # Graphs

    # Autoregression
    doublePlot(measurements.channels()[0], autoRegress[0])

    # Sliding window linear regression
    doublePlot(measurements.channels()[0], linear[0])

    # Elliptic envelope
    doublePlot(measurements.channels()[0], ret)

    # Autoregression elliptic envelope
    doublePlot(measurements.channels()[0], retElpAutoRegress)

    # Linear window elliptic envelope
    doublePlot(measurements.channels()[0], retLinRegress)

    # Sliding window mean difference + Elliptic envelope
    doublePlot(measurements.channels()[0], slidingRet)

    # Control chart
    doublePlot(measurements.channels()[0], ctrlCharted)

    # Cusum - Constant mean / std
    doublePlot(measurements.channels()[0], cusumed)

    # Cusum - Running mean / std
    doublePlot(measurements.channels()[0], runningcusumed)