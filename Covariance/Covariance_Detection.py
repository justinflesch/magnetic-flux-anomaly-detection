from configparser import Interpolation
import csv
# from curses import window
import numpy as np

import os

import logging
import matplotlib.pyplot as plt

import sys
import os

import torch


sys.path.insert(0, os.getcwd()) # set the project folder as the system folder
import data_util.ampere_data as ad

# configurations for the basic logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)


def correlationCoeffData(data_array, rowvar=False) -> np.array:
  '''
    Takes in a data array and outputs the correlation coefficient. Defaults to the columns being the indepedent variables
  
    Parameters:
      data_array: 2d numpy array

    Returns:
      C: 2d array of the correlation coefficient

    Example:
      correlationCoeffData(np.array([[0,1], [1,0]]))
  '''

  # compare each sensor
  C = np.corrcoef(data_array, rowvar=rowvar)# np.zeros( (cols, cols))
  # for i in range(cols):
  #   for j in range(i, cols):
  #     # get the correlation between theses two sensors, not itself
  #     # corrcoef should output a 2x2 matrix
  #     C[i,j] = np.corrcoef(sensors_array[:,i], sensors_array[:,j])[0,1]

  #take only a subset of the sensors (optional)


  C[C == 0] = np.nan

  return C

def plot_corrcoef(data_array, title, sensorRange=False):

  data_array = data_array[sensorRange[0][0]:sensorRange[0][1], sensorRange[1][0]: sensorRange[1][1]] if sensorRange else data_array

  cols = np.size(data_array, axis=1)

  plt.figure(figsize=(12,12))
  plt.imshow(data_array, vmin=-1.0,vmax=1.0)
  plt.grid(True, alpha=0.15)
  plt.colorbar()
  plt.yticks(np.arange(0,cols))
  plt.xticks(np.arange(0,cols))
  plt.xlabel("Sensors")
  plt.ylabel("Sensors")
  plt.title(title)
  plt.show()

  return

def plot_corrcoef_subplots(data_list, title, Nr, Nc, method=None) -> None:
  
  methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
  
  fig, axs = plt.subplots(nrows=Nr,ncols=Nc, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})
  
  # for ax, interp_method in zip(axs.flat, methods):
  for ax, x in zip(axs.flat, range(0,Nr*Nc)):
    ax.imshow(data_list[x], interpolation=method, cmap='viridis')
    ax.set_title(str(x + 1), fontsize=9, loc="center")
  plt.tight_layout()
  plt.suptitle(title, fontsize=12)
  plt.show()


def sensor_covariance(sensor1_data, sensor2_data):
  # must be the same dimensions!
  return np.corrcoef(sensor1_data, sensor2_data)[0,1]

def sensors_covariance(sensors1_data, sensors2_data, rowvar):
  # get the diagonal matrix of the sensor covariance

  # if the labels are on the columns, transpose the matrix
  # so that each array is a sensor
  if (rowvar == False):
    sensors1_data = sensors1_data.T
    sensors2_data = sensors2_data.T

  s1 = np.size(sensors1_data, axis=0)
  s2 = np.size(sensors2_data, axis=0)

  if (s1 != s2):
    print("Unequal Dim!")
  C = np.zeros((s1, s1), dtype=float)

  M = np.size(C, axis=1)
  for i in range(M):
    for j in range(i,M):
      #print("test", np.corrcoef(preds[:,i],preds[:,j]))
      C[i,j] = sensor_covariance(sensors1_data[i], sensors2_data[j])
  
  C[C == 0] = np.nan
  
  return C


if __name__ == "__main__":


  print(sys.path)
  print("ARGUMENTS PASSED:")
  for i, x in enumerate(sys.argv):
    print("\t[" + str(i) + "] " + x)

  data_list = [ad.load_data_sensors(sys.argv[x]) for x in range(1, len(sys.argv))]
  print("data_list length:", len(data_list))

  # sensors_array, sensors_rtz_array_row, sensors_rtz_array_col, row_rtz_list, col_rtz_list
  PC_list = [ad.PC_data(data) for data in data_list]
  print("PC_list length:", len(PC_list))
  print("PC_list[0] length:", len(PC_list[0]))

  subplot_row_corr_list = [sensors_covariance(PC_list[0][3][i], PC_list[1][3][i], False) for i in range(0,16)]
  subplot_col_corr_list = [sensors_covariance(PC_list[0][4][i], PC_list[1][4][i], False) for i in range(0,16)]
  
  plot_corrcoef_subplots(subplot_row_corr_list, "P-Plane Sensors", 4, 4, None)
  plot_corrcoef_subplots(subplot_col_corr_list, "C-Plane Sensors", 4, 4, None)
