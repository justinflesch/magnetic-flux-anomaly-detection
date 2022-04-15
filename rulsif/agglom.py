from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *

from sklearn.cluster import FeatureAgglomeration

from rulsif import *

import csv

# Hyperparameters
SampleWidth = 100
RetroWidth = 10
Sigma = 1 # 1 works decently
Alpha = 0.5 # 0.5 works decently
Lambda = 0.01 # 0.01 works decently

def doublePlot(data1, data2, color = None, marker = 'o'):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(len(data1)), data1)
    
    ax2.plot(range(len(data2)), data2)  
    
    ax1.set_xlabel("Sample")
    ax2.set_xlabel("Sample")
    ax2.tick_params(axis='y')

    plt.show()

def nPlot(dataArray):
    fig, ax1 = plt.subplots(1, sharex=True)
    for i in range(len(dataArray)):
        data = dataArray[i]
        ax1.plot(range(len(data)), data)
        
    ax1.set_xlabel("Sample")
    plt.show()

def quadPlot(data1, data2Tuple):
    total = np.transpose(data2Tuple[:,0,:])
    forward = np.transpose(data2Tuple[:,1,:])
    backward = np.transpose(data2Tuple[:,2,:])
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(data1, alpha=1, label="Y")
    ax2.plot(*total, alpha=1, label="Divergence")
    ax2.plot(*forward, alpha=1, label="Divergence")
    ax2.plot(*backward, alpha=1, label="Divergence")
    
    plt.show()

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)
    
    print("Data loaded")
    
    # nPlot(measurements.channels()[100:700:30])
    
    agglo = FeatureAgglomeration(n_clusters=6)
    agglo.fit(data.T)
    
    reducedData = agglo.transform(data.T).T
    
    print("Transformation complete")
    print(data.shape, reducedData.shape)
    
    # nPlot(reducedData)
    
    results = TimeSeriesDissimilarity(reducedData.T, SampleWidth, RetroWidth, Sigma, Alpha, Lambda)
    print(results.shape)

    binaryResults = results[:,0,1] > 0.95

    with open('anomalydetection.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    
        writer.writerow(results[:,0,0])
        writer.writerow(results[:,0,1])

    doublePlot(measurements.channels()[0], binaryResults)
    