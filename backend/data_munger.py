import pandas as pd
import numpy as np


def process_data(file_path):
    captions_csv = pd.read_csv(file_path)
    captions_csv.drop(columns=['Sr No'], inplace=True)
    for index, row in captions_csv.iterrows():
        if row['Caption'] is np.nan:
            captions_csv.drop(index=index, inplace=True)
        elif len(row['Caption']) > 80 or '@' in row['Caption'] or '#' in row['Caption']:
            captions_csv.drop(index=index, inplace=True)
    captions_csv.reset_index(inplace=True)
    captions_csv.drop(columns=['index'], inplace=True)
    return captions_csv


# Provide DataFrames as Parameters to the functions which you want to combine.
def concat_files(*csv_df):
    frames = list()
    for df in csv_df:
        frames.append(df)
    data = pd.concat(frames)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    data.to_csv("./data/captions_csv.csv")
