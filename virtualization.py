from nptdms import TdmsFile

def virtualize(filepath):
    tdms_file = TdmsFile.read(filepath)

    # Example of reading tdms file data.
    for group in tdms_file.groups():
        group_name = group.name
        for channel in group.channels():
            channel = channel.name
            print(group_name, channel)

if __name__ == "__main__":
    virtualize(r"C:\Users\jflesch\Capstone\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-15h-11m_SO_SGeo.tdms")