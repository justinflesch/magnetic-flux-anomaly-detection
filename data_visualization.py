import matplotlib.pyplot as plt
import numpy as np

def visualize(df):
    print("Visualizing data...")
    # rollmean = df.mean()
    # rollstd = df.std()

    #plt.figure(1)

    #print(df['/\'Measurements\'/\'Furnace Current\'']) # How to access a specific row in the dataframe.
    #print(df['/\'Measurements\'/\'P16C10z\'']) 

    for c in ['r', 't', 'z']:
        plt.figure() 

        for col in df:
            #print(col)
            if str(col)[17] == "P" and str(col)[-2] == c: # Columns that are sensor data have a P in the 17th index of the column name.
                #plt.subplot() # Add a subplot to our plot for this data.
                #print(df[col]) # Print the data within this dataframe column.
                plt.plot(df[col]) # Plot this data.
                #plt.plot(df[col].mean(), color="black", label="Rolling Mean")
                #plt.plot(rollstd[col], color="red", label="Rolling Std")
                #plt.show() # Show this given plot.

        # Fix x-axis scaling.
        xmin, xmax = plt.xlim() # Get bounds for current plot. 
        minute_labels = np.arange(0, xmax * (0.01/60), 1) # Convert ms to minutes.
        x_values = np.arange(0, len(df[col]), (len(df[col]) / len(minute_labels))) # Get which datapoint should have which minute.
        plt.xticks(x_values, np.round_(minute_labels)) # Change xticks.

        # Add labels.
        plt.title(c)
        #plt.ylabel("uT")
        plt.xlabel("time (minutes)")
        plt.legend(loc='best')

    plt.show() # Show all data in one plot.
    print("Visualization completed.")