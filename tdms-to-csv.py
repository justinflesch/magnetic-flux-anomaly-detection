# Required libraries.
from nptdms import TdmsFile
import numpy as np
import pandas as pd
import csv

# Plotting.
import matplotlib.pyplot as plt

def tdms_to_csv(filepath):
    tdms_file = TdmsFile.read(filepath) # Read in tdms file.

    df = tdms_file.as_dataframe() # Convert tdms file to Pandas dataframe.

    csv_filepath = filepath.rsplit(".", 1)[0] + ".csv" # Removed the old filetype from the fp, add the new one.

    df.to_csv(csv_filepath) # Convert df to csv file, save to passed string.

    return csv_filepath

if __name__ == "__main__":
    tdms_to_csv(r"C:\Users\jflesch\Capstone\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-14h-27m_SO_SGeo.tdms")