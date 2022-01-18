from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *

data = None
averageSamples = 100 # 100 seems to work well
anomalyThreshold = 3.0

def calculateMeansNaive(data):
    # Basic solution used for reference; seems to react more strongly to outlier datapoitns
    dims = len(data)
    samples = len(data[0])
    
    Mean = np.zeros(shape=(samples, dims))
    
    for i in range(0, samples):
        Mean[i] = np.mean(data[:, max(0, i - averageSamples):i+1], axis=1)
    
    Mean = np.transpose(Mean)
    
    return Mean

def calculateMeansConvolve(data):
    # Seems better; should be faster, seems to respond less to singular outliers
    # BUT, has bad behavior at the endpoints. I attempted to fix this below,
    # but this isn't perfect and they are still at least a bit messed up
    
    dims = len(data)
    samples = len(data[0])
    
    Mean = np.zeros(shape=(dims, samples))
    
    for i in range(dims):
        Mean[i] = np.convolve(data[i], np.ones(averageSamples) / averageSamples, mode='same')
    
    # First and Last averageSamples / 2 points are messed up; replace with original data
    errorLength = ceil(averageSamples / 2)
    
    Mean[:, 0:errorLength] = np.transpose(np.tile(Mean[:, errorLength], (errorLength, 1)))
    Mean[:, samples-errorLength:samples] = np.transpose(np.tile(Mean[:, samples - errorLength], (errorLength, 1)))
    
    return Mean

def calculateMeansCumsum(data):
    # Wrong, do not use
    dims = len(data)
    samples = len(data[0])
    print(data.shape)
    
    data = np.pad(data, averageSamples // 2)
    
    Mean = []
    for i in range(dims):
        #cumsum = np.cumsum(np.insert(data[i], 0, 0))
        #res = (cumsum[averageSamples:] - cumsum[:-averageSamples]) / float(averageSamples)
        #Mean.append(res)
        
        cumsum = np.cumsum(np.insert(data[i], 0, 0))
        cumsum[averageSamples:] = (cumsum[averageSamples:] - cumsum[:-averageSamples])
        Mean.append(cumsum[averageSamples - 1:] / float(averageSamples))
    
    Mean = np.array(Mean)
    
    print(Mean.shape)
    
    Mean = Mean[:, 0:samples]
    
    print(Mean.shape)
    
    return Mean

def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    #ax1.plot(range(len(data)), data, c=color, alpha=0.25, label="STD's from mean")
    ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    ax1.axhline(y=anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    ax1.axhline(y=-anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("STDs")
    ax1.tick_params(axis='y')
    plt.show()

def doublePlot(data1, data2, color = None, marker = 'o'):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=0.25, label="Sensor Data")
    
    ax2.scatter(list(range(len(data2))), data2, c = color, cmap = 'seismic', marker=marker)  
    ax2.axhline(y=anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    ax2.axhline(y=-anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("STDs")
    ax2.tick_params(axis='y')

    plt.show()

with TdmsFile.open("later.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    data = measurements.channels()[500:700]
    #data = measurements.channels()[500:501]
    data = np.nan_to_num(data)
    
    #plotData(data[50])
    
    print("Loading Complete")
    print("Beginning Means")
    Mean = calculateMeansConvolve(data)
    #plotData(Mean[0])  # Used for debugging convolution end behavior correction
    
    print("Means Done")
    Diff = data - Mean  # Difference from running mean for every sample in every channel
    DiffMean = np.mean(Diff, axis=0) # Average all channels together
    
    ResSTD = np.std(DiffMean)
    ResMean = np.mean(DiffMean)
    
    Res = (DiffMean - ResMean) / ResSTD
    Anomaly = (np.abs(Res) > anomalyThreshold).astype(int)
    
    print("Diff calculated")

    print("Anomaly Percentage:", np.sum(Anomaly) / len(Anomaly))

    #plotData(Res, Anomaly, 'h')
    doublePlot(data[0], Res, Anomaly, 'h')