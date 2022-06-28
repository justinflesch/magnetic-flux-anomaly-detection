from data_util.data_virtualization import virtualize
from data_util.data_normalization import normalize
from data_util.data_visualization import visualize
from data_util.ampere_data import *

if __name__ == "__main__":
    df = virtualize(r"C:\Users\fleschju\Documents\School\magnetic-flux-anomaly-detection\Test Data\VAR3---12m-15d-20y-14h-27m_SO_SGeo.tdms")
    
    normalized_df = normalize(df)
    
    visualize(normalized_df)

    