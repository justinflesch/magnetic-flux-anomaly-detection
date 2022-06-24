import matplotlib.pyplot as plt
import numpy as np
import sys
import re

from data_util.data_virtualization import virtualize

def visualize(df):
    print("Visualizing data...")
    # rollmean = df.mean()
    # rollstd = df.std()

    #plt.figure(1)

    #print(df['/\'Measurements\'/\'Furnace Current\'']) # How to access a specific row in the dataframe.
    #print(df['/\'Measurements\'/\'P16C10z\'']) 

    # for c in ['r', 't', 'z']:
    # plt.figure() 
    rtz_dict = {}

    regex = re.compile("\D\d+\D\d+(\D)")

    for col in df:
        #print(col)
        result = regex.search(col)
        if (result):
            if not result.group(1) in rtz_dict:
                rtz_dict[result.group(1)] = plt.figure(result.group(1))
            plt.figure(result.group(1)) # make label active or create new one
            #plt.subplot() # Add a subplot to our plot for this data.
            #print(df[col]) # Print the data within this dataframe column.
            plt.plot(df[col]) # Plot this data.
            #plt.plot(df[col].mean(), color="black", label="Rolling Mean")
            #plt.plot(rollstd[col], color="red", label="Rolling Std")
            #plt.show() # Show this given plot.
             # Fix x-axis scaling.
    for label, figure in rtz_dict.items():
        plt.figure(label) # activate current figure
        xmin, xmax = plt.xlim() # Get bounds for current plot. 
        minute_labels = np.arange(0, xmax * (0.01/60), 1) # Convert ms to minutes.
        x_values = np.arange(0, len(df[col]), (len(df[col]) / len(minute_labels))) # Get which datapoint should have which minute.
        plt.xticks(x_values, np.round_(minute_labels)) # Change xticks.
        plt.title(label)

        # Add labels.
        # plt.title(c)
        #plt.ylabel("uT")
        plt.xlabel("time (minutes)")
        plt.legend(loc='best')



    plt.show() # Show all data in one plot.
    print("Visualization completed.")

if __name__ == "__main__":
    print(sys.path)
    print("ARGUMENTS PASSED:")
    for i, x in enumerate(sys.argv):
        print("\t[" + str(i) + "] " + x)
    if len(sys.argv) == 2:
        
        df = virtualize(sys.argv[1])
        visualize(df)
    
    