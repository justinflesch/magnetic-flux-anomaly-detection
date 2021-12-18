import pandas as pd

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