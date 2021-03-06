import numpy as np

import sys

import logging
import matplotlib.pyplot as plt

import functools
import time

import pandas as pd

# from data_util.data_normalization import *
# from data_util.data_virtualization import *
# from data_util.data_visualization import *

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

# function wrapper for time if some function acts strangely (runs longer than time filter)
def timer_decorator(func, time_filter=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        results = func(*args, **kwargs)
        t1 = time.time()
        if (time_filter and (round((t1 -t0)*1000,1) > time_filter)) or time_filter == None:
            print('Function', func.__name__, 'time:', round((t1 -t0)*1000,1)/1000, 'seconds')
        return results
    return wrapper

# This function is expensive and should only be run once in the program! DO NOT USE
@timer_decorator
def load_data_sensors(csv_path: str) -> np.ndarray:
  """
  Loads the data from a csv file path.
  
    Parameters:
      :csv_path (String): String to file path in cwd


    Returns:
      data (np.ndarray): data with their respective labels

    Example:
      loadData_Sensors("\\VAR3 Full Melt 12-15-2020 1 Hz.csv", ((1,16),(1,16))
  """
  print(csv_path)


  data = np.genfromtxt(csv_path, dtype=np.double, delimiter=',', names=True)
  # print("TYPE:", type(data))
  # print(data)
  rows = len(data)
  cols = len(data[0])
  print("Rows and columns of data:\nrows:", rows, "cols:", cols, '\n')
  return data


# pull the MeasurementsP..C.. portion from the numpy.ndarray
# The code below needs to be refactored for pd.dataframe and torch/keras linalg
def PC_data(data, dim=((1,16),(1,16))) -> np.array:
  """
  Outputs 5 2d sensor arrays: standard, rtz magnitude by row, rtz magnitude by columns,
  List of matrices by each row, list of matrics by each col.

    Parameters:
      :data (np.ndarray): numpy data object with labl


    Returns:
      sensors_array, sensors_rtz_array_row, sensors_rtz_array_col, row_rtz_list, col_rtz_list

    Example:
      loadData_Sensors("\\VAR3 Full Melt 12-15-2020 1 Hz.csv", ((1,16),(1,16))
  """
  l1 = ['r', 't', 'z']
   # list of strings for the PC sensors labels (not necesarily in our data)
  sensors_row = np.array(["MeasurementsP" + f'{x:02d}' + "C" + f'{y:02d}' + z for x in range(dim[0][0], dim[0][1] + 1) for y in range(dim[1][0], dim[1][1] + 1) for z in l1])
  num_sensors = np.size(sensors_row)
  # reorder the sensors to be grouped by columns now
  sensors_col = np.array(["MeasurementsP" + f'{x:02d}' + "C" + f'{y:02d}' + z for y in range(dim[1][0], dim[1][1] + 1) for x in range(dim[0][0], dim[0][1] + 1) for z in l1])

  print("The number of total possible PC sensors:\n", num_sensors)

  # reshaped for rtz
  sensors_rtz = sensors_row.reshape(int(num_sensors/3), 3)
  # order the sensors by the columns
  sensors_rtz2 = sensors_col.reshape(int(num_sensors/3), 3)
  sensors_row_rtz = sensors_row.reshape(int(num_sensors/(3*16)), 16, 3)
  sensors_col_rtz = sensors_col.reshape(int(num_sensors/(3*16)), 16, 3)


  # get all the sensors in the sensor array that exists in the data
  sensors_array = np.transpose(np.array([data[x] for x in sensors_row if x in data.dtype.fields]).astype(np.double))
  rows = np.size(sensors_array, axis=0)
  cols = np.size(sensors_array, axis=1)
  print("Rows and columns of PC sensors array:\nrows:", rows, "cols:", cols, '\n')

  # make sure for each magnitude that the r,t,z values exists for each of them
  # get only the magnitude if there exists an rtz value for that specific sensor
  # transpose the matrix so that the columns are the sensors and the rows are the time
  sensors_rtz_array_row = np.transpose(np.array([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in sensors_rtz \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]).astype(np.double))

  sensors_rtz_array_col = np.transpose(np.array([np.linalg.norm([data[rtz[0]], data[rtz[1]], data[rtz[2]]], axis=0) for rtz in sensors_rtz2 \
  if rtz[0] in data.dtype.fields and rtz[1] in data.dtype.fields and rtz[2] in data.dtype.fields]).astype(np.double))

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
  print("Rows and columns of PC sensor rtz magnitude array:\nrows:", rows_mag, "cols_mag:", cols_mag, '\n')

  # Save our data so we can look at it (if need be)
  # np.savetxt("sensor_array.csv", sensors_array, delimiter=',')
  # np.savetxt("sensor_rtz_array.csv", sensors_rtz_array_row, delimiter=',')



  return sensors_array, sensors_rtz_array_row, sensors_rtz_array_col, row_rtz_list, col_rtz_list
  
def PC_data_pandas(df: pd.DataFrame) -> pd.DataFrame:

  # print("MISSING VALUES:\n", df.isnull())

  # print("SUM OF MISISNG VALUES:\n", df.isnull().sum())

  # print("ROWS ARE NULL:\n", df[df.isnull().any(axis=1)])
  # print("COLS ARE NULL:\n", df.columns[df.isnull().any()].tolist())

  # Drop the columns that contain null values
  df = df.dropna(axis=1, how='any')


  # create our labels to add to the pandas frame
  rtz_list = ["MeasurementsP" + f'{x:02d}' + "C" + f'{y:02d}_rtz' for x in range(1,17) for y in range(1,17) \
  if "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "r\'" in df \
  and "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "t\'" in df \
  and "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "z\'" in df]
  # print("yes" if "/\'Measurements\'/\'P01C01r\'" in df else "no")
  # print(rtz_list)

  # create a tensor directly from data
  n_array = np.array([[df["/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "r\'"], df["/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "t\'"], df["/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "z\'"]] \
  for x in range (1,17) for y in range (1,17) \
  if "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "r\'" in df \
  and "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "t\'" in df \
  and "/\'Measurements\'/\'P" + f'{x:02d}' + "C" + f'{y:02d}' + "z\'" in df])
  # print(n_array)
  t_array = torch.from_numpy(n_array)
  # print(t_array)
  
  t_array.to("cuda") if torch.cuda.is_available() else t_array.to("cpu")

  rtz_tensor = torch.linalg.vector_norm(t_array, ord=2, dim=1)


  # print(f"Shape of rtz_tensor: {rtz_tensor.shape}")
  # print(f"Datatype of rtz_tensor: {rtz_tensor.dtype}")
  # print(f"Device rtz_tensor is stored on: {rtz_tensor.device}")

  transpose_rtz_tensor = rtz_tensor.T

  # print(f"Shape of transpose_rtz_tensor: {transpose_rtz_tensor.shape}")
  # print(f"Datatype of transpose_rtz_tensor: {transpose_rtz_tensor.dtype}")
  # print(f"Device transpose_rtz_tensor is stored on: {transpose_rtz_tensor.device}")


  df2 = pd.DataFrame(transpose_rtz_tensor.numpy(), columns=rtz_list)
  # print(df2)
  result = pd.concat([df,df2], axis=1)

  # print(result)

  return result



# graph the subset labels from the np.ndarray
@timer_decorator
def graph_labels(data, labels) -> None:

  """
  Outputs a graph based on a list of labels provided

    Parameters:
      data (np.ndarray): numpy data object with labl


    Returns:
      None

    Example:
      graph_labels(data, ["MeasurementsCurrent"])
  """
  color = 'tab:blue'

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

@timer_decorator
def compare_labels(data1, data2, labels, path_list=None) -> None:
  color = 'tab:blue'

  data1 = [data1[label] for label in labels]

  fig, ax1 = plt.subplots(figsize=(16,9))

  ax1.plot(range(len(data1[0])), data1[0], c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.set_ylabel(labels[0])
  ax1.tick_params(axis='y', labelcolor=color)

  for i in range(1, len(data1)):
    ax2 = ax1.twinx()
    ax2.plot(range(len(data1[i])), data1[i], c=color, alpha=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[i])
    ax2.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'

  data2 = [data2[label] for label in labels]

  ax1.plot(range(len(data2[0])), data2[0], c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.set_ylabel(labels[0])
  ax1.tick_params(axis='y', labelcolor=color)

  for i in range(1, len(data2)):
    ax2 = ax1.twinx()
    ax2.plot(range(len(data2[i])), data2[i], c=color, alpha=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[i])
    ax2.tick_params(axis='y', labelcolor=color)

  if path_list:
    for tuple in path_list:
      plt.plot([tuple[0], tuple[1]], [data2[0][tuple[0]], data2[0][tuple[1]]], 'k-', lw=1)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()

@timer_decorator
def compare_data(data1, data2, path_list=None) -> None:
  color = 'tab:blue'

  fig, ax1 = plt.subplots(figsize=(16,9))

  ax1.plot(range(len(data1)), data1, c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'


  ax1.plot(range(len(data2)), data2, c=color, alpha=1)
  ax1.set_xlabel("Time")
  ax1.tick_params(axis='y', labelcolor=color)

  if path_list:
    for tuple in path_list:
      plt.plot([tuple[0], tuple[1]], [data1[tuple[0]], data2[tuple[1]]], 'k-', lw=0.5)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()




if __name__ == "__main__":
  print("ARGUMENTS PASSED:")
  for i, x in enumerate(sys.argv):
    print("\t[" + str(i) + "] " + x)
  # passed arguments start at index 1
  # print("Loading csv data...")
  # # data_list = [load_data_sensors(sys.argv[x]) for x in range(1, len(sys.argv))]
  # data_list = load_data_sensors("Capstone Data\\Type A\\Combined---EQS-VAR3---09m-09d-20y_1s.csv")
  # print("Finished laoding csv data :)")
  # data_list2 = load_data_sensors_pandas("Capstone Data\\Type A\\Combined---EQS-VAR3---09m-09d-20y_1s.tdms")
  # print("Finished laoding tdms data :)")

  # print("Pulling PC data...")
  # PC_list = [[PC_data(data)] for data in data_list]
  # print("Finished pulling and processing PC data :)")

  # print("Graphing current in first data")
  # graph_labels(data_list[0], ["MeasurementsCurrent"])
  # print("Finished graphing first data\'s current :)")

  # test = np.array([
  #   [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
  #   [[1,2,3,4], [1,2,3,4], [1,2,3,4]]],
  #   dtype=np.float
  # )
  # print(np.linalg.norm(test, axis=1))

  # test2 = torch.from_numpy(test)

  # print(torch.linalg.vector_norm(test2, ord=2, dim=1))

  # print(test)
  from data_virtualization import virtualize
  from data_normalization import normalize_min_max
  df = virtualize(sys.argv[1])

  # print(df)

  df_rtz = PC_data_pandas(df)
  df_rtz_normalized = normalize_min_max(df_rtz)
  print(df_rtz_normalized)