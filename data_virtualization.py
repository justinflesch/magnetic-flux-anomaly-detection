# Required libraries.
from nptdms import TdmsFile

def virtualize(filepath):
    tdms_file = TdmsFile.read(filepath) # Read in tdms file.

    df = tdms_file.as_dataframe() # Convert tdms file to Pandas dataframe.
    return df


if __name__ == "__main__":
    virtualize(r"C:\Users\jflesch\Capstone\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-15h-11m_SO_SGeo.tdms")