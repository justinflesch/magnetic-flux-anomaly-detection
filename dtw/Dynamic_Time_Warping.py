from configparser import Interpolation
import numpy as np

import sys

import logging
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import os

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


def fast_dtw_label(data1, data2, label):


  data_label1 = data1[label]
  data_label2 = data2[label]

  distance = dtw.distance_fast(data_label1, data_label2)
  print("DISTANCE:", distance)
  distance, paths = dtw.warping_paths_fast(data_label1, data_label2)
  print("DISTANCE:", distance, "PATHS", paths)
  path = dtw.warping_path_fast(data_label1, data_label2)
  return path

# iterating through the path_mapping to create two similar graphs
def transform_graphs_from_dtw_path(data1, data2, path_mapping, label):

  new_data1 = np.array([data1[label][map[0]] for map in path_mapping])
  new_data2 = np.array([data2[label][map[1]] for map in path_mapping])


  return new_data1, new_data2




if __name__ == "__main__":
  print("ARGUMENTS PASSED:")
  for i, x in enumerate(sys.argv):
    print("\t[" + str(i) + "] " + x)

  data_list = [ad.load_data_sensors(sys.argv[x]) for x in range(1, len(sys.argv))]

  print("calculating dtw...")
  path = fast_dtw_label(data_list[0], data_list[1], "MeasurementsCurrent")
  print("Finished calculating dtw path :)")

  path_vis = path[::200]
  ad.compare_data(data_list[0]["MeasurementsCurrent"], data_list[1]["MeasurementsCurrent"], path_vis)

  new_data1, new_data2 = transform_graphs_from_dtw_path(data_list[0], data_list[1], path, "MeasurementsCurrent")
  print(type(new_data1), type(new_data2), len(new_data1), len(new_data2))
  ad.compare_data(new_data1, new_data2)




