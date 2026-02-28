import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_rul(df):
    # for adding rul column

    max_cycle = df.groupby("unit_number")["time_in_cycles"].max()
    df["RUL"] = df.apply(
            lambda row: max_cycle[row["unit_number"]] - row["time_in_cycles"], axis=1
            )
    return df

def remove_constant_sensors(df):
    #constant sensors have zero variance and hence did not contribute to prediction

    sensor_cols = [col for col in df.columns if "sensor" in col]
    variances = df[sensor_cols].var()
    constant_sensors = variances[variances == 0.0].index.tolist()
    print("Removed sensors with zero variances : ", constant_sensors)

    df = df.drop(columns=constant_sensors)
    return df

def scale_sensors(df):
    #standardization of sensor features
    sensor_cols = [col for col in df.columns if "sensor" in col]
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, scaler

def preprocess_pipeline(df):
    print("checking NaNs: ", df.isnull().sum().sum())
    df = add_rul(df)
    df = remove_constant_sensors(df)
    df, scaler = scale_sensors(df)

    return df, scaler


