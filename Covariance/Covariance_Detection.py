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
      data_array (2d numpy array): the data array to correlate
      rowvar (boolean): set if the rows are the variables

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

def plot_corrcoef(corrcoef_array, title, sensorRange=False) -> None:
  ''' 
  Plots a single correlation coefficient with a range of data to correlate

    Parameters:
      corrcoef_array (np.array): correlation coefficient array
      title (string): title of the graph
      sensorRange (list): a list of tiples signifying the range of data to use

    Returns: 
      None

    Example:
      plot_corrcoef(corrcoef_array, "Correlation Coefficient", [(0,16), (0,16)])
  '''

  corrcoef_array = corrcoef_array[sensorRange[0][0]:sensorRange[0][1], sensorRange[1][0]:sensorRange[1][1]] if sensorRange else corrcoef_array

  cols = np.size(corrcoef_array, axis=1)

  plt.figure(figsize=(12,12))
  plt.imshow(corrcoef_array, vmin=-1.0,vmax=1.0)
  plt.grid(True, alpha=0.15)
  plt.colorbar()
  plt.yticks(np.arange(0,cols))
  plt.xticks(np.arange(0,cols))
  plt.xlabel("Sensors")
  plt.ylabel("Sensors")
  plt.title(title)
  plt.show()

  return


def plot_corrcoef_subplots(corrcoef_list, title, Nr, Nc, method=None) -> None:

  ''' 
  Plots subplots with the visual dimnesions Nr and Nc using a method of your choice

    Parameters:
      corrcoef_list (list): a list of np.array coefficient data to plot
      title (string): title for the plot
      Nr (int): the row dimension for the visual plot
      Nc (int): the column dimension for the visual plot
      method: the interpolation method for the correlation plot. Methods are shown in the function
    
    Return:
      None 
  '''
  # methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
  #          'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
  #          'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
  
  fig, axs = plt.subplots(nrows=Nr,ncols=Nc, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})
  
  # for ax, interp_method in zip(axs.flat, methods):
  for ax, x in zip(axs.flat, range(0,Nr*Nc)):
    ax.imshow(corrcoef_list[x], interpolation=method, cmap='viridis')
    ax.set_title(str(x + 1), fontsize=9, loc="center")
  plt.tight_layout()
  plt.suptitle(title, fontsize=12)
  plt.show()


def sensor_covariance(sensor1_data, sensor2_data) -> np.float:
  '''
  Return a np.array of sensor correlation of two 1d arrays. The inputs must be the same dimensions.
    
    Parameters:
      sensor1_data (np.array): 1-d array to correlate
      sensor2_data (np.array): 1-d array to correlate

    Return:
      np.float: correlation of two 1d arrays
  '''
  if (sensor1_data.size != sensor2_data.size):
    raise Exception("sensors_covariance: unequal input dimensions.")
  # must be the same dimensions!
  return np.corrcoef(sensor1_data, sensor2_data)[0,1]

def sensors_covariance(sensors1_data, sensors2_data, rowvar) -> np.array:
  '''
  Takes in two 2d arrays of data and returns a diagonal correlation matrix

    Parameters:
      sensor1_data (np.array): 2d array of sensor 1 data
      sensor2_data (np.array): 2d array of sensor 2 data
      rowvar (boolean): boolean to transpose data if the variables are on the rows
  '''
  # get the diagonal matrix of the sensor covariance

  # if the labels are on the columns, transpose the matrix
  # so that each array is a sensor
  if (rowvar == False):
    sensors1_data = sensors1_data.T
    sensors2_data = sensors2_data.T

  s1 = np.size(sensors1_data, axis=0)
  s2 = np.size(sensors2_data, axis=0)

  if (s1 != s2):
    raise Exception("sensors_covariance: unequal input dimensions.")
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
