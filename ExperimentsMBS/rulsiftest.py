import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from math import *
from rulsif import *


SampleWidth = 35
RetroWidth = 10
Sigma = 8 # I have no idea what this should be
Alpha = 0.5 # Also no real clue here
Lambda = 0.01


def plotData(data, color = None, marker = 'o'):
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.plot(range(len(data)), data, c=color, alpha=0.25, label="STD's from mean")
    #ax1.scatter(list(range(len(data))), data, c = color, cmap = 'seismic', marker=marker)  
    #ax1.axhline(y=anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    #ax1.axhline(y=-anomalyThreshold, color='r', linestyle=(0, (5, 10)))
    
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("STDs")
    ax1.tick_params(axis='y')
    plt.show()

def doublePlot(data1, data2):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(range(len(data1)), data1, alpha=0.25, label="Y")
    ax2.plot(range(len(data2)), data2, alpha=0.25, label="Divergence")
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

# Test 0: 3 Normal Distributions (Figure 3 from paper)
dataset = np.zeros(600)
for i in range(0,200):
    dataset[i] = np.random.normal(0,4)
for i in range(200,400):
    dataset[i] = np.random.normal(0,1)
for i in range(400,600):
    dataset[i] = np.random.normal(0,4)

dataset = dataset[..., np.newaxis]

results = TimeSeriesDissimilarity(dataset, 25, 5, 8, Alpha, 0.05)
#results = TimeSeriesDissimilarityStepped(dataset, 10, 25, 5, 8, Alpha, 0.05)
quadPlot(dataset, results, [200,400])

# Test 1: Jumping Means
dataset = np.zeros(5000)
mu = 0
Nprev = 0
for i in range(0,5000):
    N = ceil(i / 100)
    if N != Nprev:
        Nprev = N
        mu += N / 16
    
    if i == 1 or i == 2:
        dataset[i] = 0
    else:
        dataset[i] = 0.6*dataset[i-1] - 0.5*dataset[i-2] + np.random.normal(mu, 1.5)

dataset = dataset[..., np.newaxis]

#plotData(dataset)
results = TimeSeriesDissimilarity(dataset, SampleWidth, RetroWidth, Sigma, Alpha, Lambda)
#plotData(results)
quadPlot(dataset, results, range(100,5000,100))