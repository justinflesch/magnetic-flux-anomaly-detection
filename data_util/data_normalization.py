import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_virtualization import virtualize

import sys

def normalize(df):
    print("Beginning data normalization...")
    normalized_df = df.copy()

    # Min-Max Method.  Normalizes values to be between 0 and 1.
    # for col in normalized_df.columns:
    #     normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    # return normalized_df

    # Maximum Absolute Scaling Method.  Normalizes values to be between -1 and 1, mean of 0, standard deviation of 1.
    for col in normalized_df.columns:
        normalized_df[col] = normalized_df[col] / normalized_df[col].abs().max()
    print("Data normalization completed.")

    return normalized_df

def normalize_min_max(df, labels, normalize_labels):
    scaler = MinMaxScaler()

    df_normalized = df[labels].copy()

    df_normalized[normalize_labels] = scaler.fit_transform(df_normalized[normalize_labels])

    return df_normalized

if __name__ == "__main__":
    print(sys.path)
    print("ARGUMENTS PASSED:")
    for i, x in enumerate(sys.argv):
        print("\t[" + str(i) + "] " + x)
    if len(sys.argv) == 2:
        
        df = virtualize(sys.argv[1])
        print(df)
        df_normalized = normalize_min_max(df, ["/'Measurements'/'Current'"], ["/'Measurements'/'Current'"])

        print(df_normalized)