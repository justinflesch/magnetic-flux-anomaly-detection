# Required libraries.
from nptdms import TdmsFile
import numpy as np
import pandas as pd

# Bokeh plots for viewing data.
import matplotlib.pyplot as plt

def virtualize(filepath):
    tdms_file = TdmsFile.read(filepath) # Read in tdms file.

    df = tdms_file.as_dataframe() # Convert tdms file to Pandas dataframe.
    print(df['/\'Measurements\'/\'P16C10z\''])

    # Resample the dataset by averages.
    rollmean = df.mean()
    rollstd = df.std()

    plt.figure(1)

    #print(df['/\'Measurements\'/\'Furnace Current\'']) # How to access a specific row in the dataframe.
    #print(df['/\'Measurements\'/\'P16C10z\'']) 
    
    for col in df:
        print(col)
        if str(col)[17] == "P": # Columns that are sensor data have a P in the 17th index of the column name.
            plt.subplot() # Add a subplot to our plot for this data.
            #print(df[col]) # Print the data within this dataframe column.
            plt.plot(df[col], label="Raw Data") # Plot this data.
            plt.plot(df[col].mean(), color="black", label="Rolling Mean")
            plt.plot(rollstd[col], color="red", label="Rolling Std")
            plt.legend(loc='best')
            plt.title(col)
            plt.show() # Show this given plot.

    # plt.show() # Show all data in one plot.


if __name__ == "__main__":
    virtualize(r"C:\Users\jflesch\Capstone\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-15h-11m_SO_SGeo.tdms")