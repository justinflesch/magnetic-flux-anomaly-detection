# Required libraries.
from nptdms import TdmsFile
import numpy as np
import pandas as pd

# Bokeh plots for viewing data.
import matplotlib.pyplot as plt

def virtualize(filepath):
    tdms_file = TdmsFile.read(filepath)

    df = tdms_file.as_dataframe()
    print(df)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(grp_data)
    plt.show()


if __name__ == "__main__":
    virtualize(r"C:\Users\jflesch\Capstone\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-15h-11m_SO_SGeo.tdms")