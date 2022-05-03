from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *

from sklearn.cluster import FeatureAgglomeration

from rulsif import *

# Hyperparameters
SampleWidth = 100
RetroWidth = 10
Sigma = 1.5 # 1 works decently, also 4
Alpha = 0.5 # 0.5 works decently
Lambda = 0.01 # 0.01 works decently

def triplePlot(current, data, scores):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    
    ax1.plot(range(len(current)), current)
    
    for i in range(len(data)):
        subData = data[i]
        ax2.plot(range(len(subData)), subData)
    
    ax3.plot(range(len(scores)), scores)      
     
    ax1.set_xlabel("Sample")
    plt.show()

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
    ax1.plot(data1, alpha=0.25, label="Y")
    ax2.plot(*total, alpha=0.25, label="Divergence")
    ax2.plot(*forward, alpha=0.25, label="Divergence")
    ax2.plot(*backward, alpha=0.25, label="Divergence")
    
    plt.show()

def getAnomalyScores(length):
    result = np.zeros(length)
    
    result[1480:1580] = 1
    result[2500:7000] = 1
    
    return result

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)
    
    print("Data loaded")
    
    agglo = FeatureAgglomeration(n_clusters=6)
    agglo.fit(data.T)
    
    reducedData = agglo.transform(data.T).T
    
    print("Transformation complete")
    print(data.shape, reducedData.shape)

    results = TimeSeriesDissimilarity(reducedData.T, SampleWidth, RetroWidth, Sigma, Alpha, Lambda)
    resultsTotal = results[:,0,:]
    
    print(results.shape)
    #quadPlot(measurements.channels()[0], results)
    triplePlot(measurements.channels()[0], reducedData, resultsTotal[:,1])
    
    
    print(resultsTotal.shape)
    
    doublePlot(measurements.channels()[0], resultsTotal[:,1] > 0.01)
    
    while True:
        try:
            np.savetxt("output.csv", resultsTotal.T, delimiter=",", fmt='%f')
        except PermissionError:
            input("write failed; press enter to retry")
        else:
            break
    
    # Calculate ROC curve
    
    # 0 is true positive
    # 1 is false positive
    results = np.zeros((101, 2))
    
    trueResults = getAnomalyScores(resultsTotal[:,1].shape[0])
    
    for i in range (0,101):
        threshold = i * 0.01
        
        anomalies = resultsTotal[:,1] > threshold
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        for j in range(0, anomalies.shape[0]):
            """
            trueResult = trueResults[int(resultsTotal[j, 0]) - 1]
            anomResult = anomalies[j]
            
            if   trueResult == 1 and anomResult == 1:
                TP += 1
            elif trueResult == 1 and anomResult == 0:
                FN += 1
            elif trueResult == 0 and anomResult == 0:
                TN += 1
            elif trueResult == 0 and anomResult == 1:
                FP += 1
            """
            
            if trueResults[j] == 1 and anomalies[j] == 1:
                TP += 1
            elif trueResults[j] == 1 and anomalies[j] == 0:
                FN += 1
            elif trueResults[j] == 0 and anomalies[j] == 0:
                TN += 1
            elif trueResults[j] == 0 and anomalies[j] == 1:
                FP += 1
            
        
        results[i][1] = TP / (TP + FN)
        results[i][0] = FP / (FP + TN)
        
        
    
    fig, ax1 = plt.subplots(1, sharex=True)
    
    ax1.plot(results[:,0].tolist(), results[:,1].tolist())
    
    for i in range(0, results.shape[0]):
        threshold = i * 0.01
        
        ax1.annotate(threshold, (results[i, 0], results[i, 1]))
    
    ax1.set_xlabel("Sample")

    plt.show()
    
    
    
    