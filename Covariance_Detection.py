from configparser import Interpolation
import csv
import numpy as np

import os

import logging
import matplotlib.pyplot as plt

import torch


# configurations for the basic logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

def loadData_Sensors(csv_path, dim) -> np.array:
  '''
  Loads the data from a csv file path. Outputs 5 2d sensor arrays: standard, rtz magnitude by row, rtz magnitude by columns,
  List of matrices by each row, list of matrics by each col.
  
    Parameters:
      csv_path (String): String to file path in cwd
      dim (Tuple): Dimensions of the data Tuple, starting with the first numerical string and ending with last

    Returns:
      sensors_array (np.array): Five numpy arrays

    Example:
      loadData_Sensors("\\VAR3 Full Melt 12-15-2020 1 Hz.csv", ((1,16),(1,16))
  '''
  cwd = os.getcwd()
  print(cwd + csv_path)

  l1 = ['r', 't', 'z']

  data = np.genfromtxt(cwd + csv_path, dtype=None, delimiter=',', names=True)

  # print(data)
  rows = len(data)
  cols = len(data[0])
  print("Rows and columns of data:\nrows:", rows, "cols:", cols, '\n')

  # list of strings for the sensor labels (not necesarily in our data)
  sensors_row = np.array(["MeasurementsP" + f'{x:02d}' + "C" + f'{y:02d}' + z for x in range(dim[0][0], dim[0][1] + 1) for y in range(dim[1][0], dim[1][1] + 1) for z in l1])
  num_sensors = np.size(sensors_row)
  # reorder the sensors to be grouped by columns now
  sensors_col = np.array(["MeasurementsP" + f'{x:02d}' + "C" + f'{y:02d}' + z for y in range(dim[1][0], dim[1][1] + 1) for x in range(dim[0][0], dim[0][1] + 1) for z in l1])

  print("The number of total possible sensors:\n", num_sensors)

  # reshaped for rtz
  sensors_rtz = sensors_row.reshape(int(num_sensors/3), 3)
  # order the sensors by the columns
  sensors_rtz2 = sensors_col.reshape(int(num_sensors/3), 3)
  sensors_row_rtz = sensors_row.reshape(int(num_sensors/(3*16)), 16, 3)
  sensors_col_rtz = sensors_col.reshape(int(num_sensors/(3*16)), 16, 3)
  # print(sensors_rtz)
  # print(sensors_col_rtz)

  # get all the sensors in the sensor array that exists in the data
  sensors_array = np.transpose(np.array([data[x] for x in sensors_row if x in data.dtype.fields]).astype(float))
  rows = np.size(sensors_array, axis=0)
  cols = np.size(sensors_array, axis=1)
  print("Rows and columns of sensors array:\nrows:", rows, "cols:", cols, '\n')

  # make sure for each magnitude that the r,t,z values exists for each of them
  # get only the magnitude if there exists an rtz value for that specific sensor
  # transpose the matrix so that the columns are the sensors and the rows are the time
  sensors_rtz_array_row = np.transpose(np.array([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in sensors_rtz \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]).astype(float))

  sensors_rtz_array_col = np.transpose(np.array([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in sensors_rtz2 \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]).astype(float))

  # each row has their own respective matrix
  row_rtz_list = [np.transpose([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in P \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]) for P in sensors_row_rtz]

  # each column has their own respective matrix
  col_rtz_list = [np.transpose([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in P \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]) for P in sensors_col_rtz]
  # reshape each of these columns so that the rtz gets their own respective rtz

  # it should print roughly the sqrt(r^2, t^2, z^2) for each sensor. Double check for first sensor
  # print(sensors_rtz_array_row)
  # print(sensors_array)
  rows_mag = np.size(sensors_rtz_array_row, axis=0)
  cols_mag = np.size(sensors_rtz_array_row, axis=1)

  # it should be about 1/3 the number of columns from the original sensors_array
  print("Rows and columns of sensor rtz magnitude array:\nrows:", rows_mag, "cols_mag:", cols_mag, '\n')

  # Save our data so we can look at it (if need be)
  # np.savetxt("sensor_array.csv", sensors_array, delimiter=',')
  # np.savetxt("sensor_rtz_array.csv", sensors_rtz_array_row, delimiter=',')



  return sensors_array, sensors_rtz_array_row, sensors_rtz_array_col, row_rtz_list, col_rtz_list

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
  plt.imshow(data_array, vmin=0,vmax=1)
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


def load_and_graph_label(csv_path, labels) -> None:

  color = 'tab:blue'

  cwd = os.getcwd()
  data = np.genfromtxt(cwd + csv_path, dtype=None, delimiter=',', names=True)
  print(data.dtype)

  data_list = [data[label] for label in labels]

  fig, ax1 = plt.subplots(figsize=(16,9))

  ax1.plot(range(len(data_list[0])), data_list[0], c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.set_ylabel(labels[0])
  ax1.tick_params(axis='y', labelcolor=color)

  for i in range(1, len(data_list)):
    ax2 = ax1.twinx()
    ax2.plot(range(len(data_list[i])), data_list[i], c=color, alpha=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[i])
    ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()

def load_and_compare_labels(csv_path1, csv_path2, labels):
  color = 'tab:blue'

  cwd = os.getcwd()
  data = np.genfromtxt(cwd + csv_path1, dtype=None, delimiter=',', names=True)
  print(data.dtype)

  data_list = [data[label] for label in labels]

  fig, ax1 = plt.subplots(figsize=(16,9))

  ax1.plot(range(len(data_list[0])), data_list[0], c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.set_ylabel(labels[0])
  ax1.tick_params(axis='y', labelcolor=color)

  for i in range(1, len(data_list)):
    ax2 = ax1.twinx()
    ax2.plot(range(len(data_list[i])), data_list[i], c=color, alpha=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[i])
    ax2.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'

  cwd = os.getcwd()
  data = np.genfromtxt(cwd + csv_path2, dtype=None, delimiter=',', names=True)
  print(data.dtype)

  ax1.plot(range(len(data_list[0])), data_list[0], c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.set_ylabel(labels[0])
  ax1.tick_params(axis='y', labelcolor=color)

  for i in range(1, len(data_list)):
    ax2 = ax1.twinx()
    ax2.plot(range(len(data_list[i])), data_list[i], c=color, alpha=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[i])
    ax2.tick_params(axis='y', labelcolor=color)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
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

  # load_and_compare_labels("\\Combined---EQS-VAR3---09m-01d-20y_1s.csv", "\\Combined---EQS-VAR3---09m-02d-20y_1s.csv", ["MeasurementsPower"])

  data1, row_data, col_data, row_list, col_list = loadData_Sensors("\\VAR3 Full Melt 12-15-2020 1 Hz.csv", ((1,16), (1,16)))
  # plot_corrcoef(correlationCoeffData(data1), "Sensor Array (sorted by P-plane)")
  # plot_corrcoef(correlationCoeffData(row_data), "sensor Magnitude Array (sorted by P-Plane)")
  # plot_corrcoef(correlationCoeffData(col_data), "sensor Magnitude Array (sorted by C-Plane)")

  # plot_corrcoef(sensors_covariance(row_data, row_data, False), "Compare sensor covariance (row vs. col")

  # subplot_row_corr_list = [correlationCoeffData(array) for array in row_list]
  # subplot_col_corr_list = [correlationCoeffData(array) for array in col_list]

  # plot_corrcoef_subplots(subplot_row_corr_list, "P-Plane Sensors", 4, 4, None)
  # plot_corrcoef_subplots(subplot_col_corr_list, "C-Plane Sensors", 4, 4, None)

  # compare with itself for testing
  subplot_row_corr_list = [sensors_covariance(array,array, False) for array in row_list]
  subplot_col_corr_list = [sensors_covariance(array,array, False) for array in col_list]
  plot_corrcoef_subplots(subplot_row_corr_list, "P-Plane Sensors", 4, 4, None)
  plot_corrcoef_subplots(subplot_col_corr_list, "C-Plane Sensors", 4, 4, None)




