# Required libraries.
from nptdms import TdmsFile
import numpy as np
import pandas as pd
import csv
import os
import sys

# Plotting.
import matplotlib.pyplot as plt

def tdms_to_csv(filepath):
    tdms_file = TdmsFile.read(filepath) # Read in tdms file.

    df = tdms_file.as_dataframe() # Convert tdms file to Pandas dataframe.

    csv_filepath = filepath.rsplit(".", 1)[0] + ".csv" # Removed the old filetype from the fp, add the new one.

    df.to_csv(csv_filepath) # Convert df to csv file, save to passed string.

    with open(csv_filepath, 'r') as original: data = original.read()
    with open(csv_filepath, 'w') as modified: modified.write("/\'Time\'" + data)

    return csv_filepath

# convert the files in directory and subdirectories tdms
def tdms_to_csv_dir(dir_path, replace=False):
    for subdir, dirs, files in os.walk(dir_path):
        print(subdir, dirs, files)
        # for dir in dirs:
        #     # recursive call
        #     print("Subdirectory prefix: ", subdir)
        #     print("Directories to search: ", dirs)
        #     # tdms_to_csv_dir(os.path.join(subdir, dir))
        for file in files:
            if (file.endswith('.tdms') and ((file.replace(".tdms", ".csv") not in files) or replace == True)):
                tdms_to_csv(os.path.join(subdir, file))


if __name__ == "__main__":
    if len(sys.argv) == 2: # if we only have one argument passed
        tdms_to_csv_dir(sys.argv[1], True)
    elif len(sys.argv) == 3: # take in an a boolean argument
        if sys.argv[2] == "True" or sys.argv[2] == "true":
            tdms_to_csv_dir(sys.argv[1], True)
        else:
            tdms_to_csv_dir(sys.argv[1], False)