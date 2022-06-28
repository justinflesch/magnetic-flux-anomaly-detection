import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# from data_util.data_virtualization import virtualize

import sys

def normalize(df: pd.DataFrame) -> pd.DataFrame:
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

def normalize_min_max(df: pd.DataFrame, labels: list=None, normalize_labels: list=None):
    scaler = MinMaxScaler()
    if labels:
        df = df[labels].copy()
    if normalize_labels:
        df[normalize_labels] = scaler.fit_transform(df[normalize_labels])
        return df
    else:
        df[df.columns] = scaler.fit_transform(df[df.columns])
        return df

if __name__ == "__main__":
    from data_virtualization import virtualize
    print(sys.path)
    print("ARGUMENTS PASSED:")
    for i, x in enumerate(sys.argv):
        print("\t[" + str(i) + "] " + x)
    if len(sys.argv) == 2:
        
        df = virtualize(sys.argv[1])
        print(df)
        df_normalized = normalize_min_max(df, ["/'Measurements'/'Current'"], ["/'Measurements'/'Current'"])

        print(df_normalized)