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

def ctrlChart(data, stdMult = 2, sndAlarm = 0.25):
    ctrl = np.zeros(data.shape[1])

    # Going through each channel for 1st level alarm
    for channel in range(0, data.shape[0]):
        # Initializing running mean / standard deviation
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

with TdmsFile.open("fullmelt - 0.tdms") as tdms_file:
    all_groups = tdms_file.groups()
    measurements = tdms_file['Measurements']
    
    #data = measurements.channels()[500:510]
    data = measurements.channels()[100:700:4]
    
    data = np.nan_to_num(data)

    ctrlCharted = ctrlChart(data, 3.5, 0.35)

    doublePlot(measurements.channels()[0], ctrlCharted)