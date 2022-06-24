# Required libraries.
from nptdms import TdmsFile
import sys

def virtualize(filepath):
    tdms_file = TdmsFile.read(filepath) # Read in tdms file.

    df = tdms_file.as_dataframe() # Convert tdms file to Pandas dataframe.
    return df


if __name__ == "__main__":
    print(sys.path)
    print("ARGUMENTS PASSED:")
    for i, x in enumerate(sys.argv):
        print("\t[" + str(i) + "] " + x)
    if len(sys.argv) == 2:
        
        df = virtualize(sys.argv[1])
        print(df)