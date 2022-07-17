import pandas as pd
import numpy as np

non_alpha_num = ['!', '@', '#', '$', '%', '&', '*', '"', "'", ".", ",", '?', ';', ":"]

def addCaptionTags(captions):
    caps = []
    for txt in captions:
        txt = '<START>' + txt + "<END>"
        caps.append(txt)
    return caps
# Provide file paths as parameter.
# Returns a DataFrame with all the image locations and captions
def process_data(*file_paths):
    # global non_alpha_num
    # df = {"image": list(),
    #       "caption": list()}
    # for file_path in file_paths:
    #     captions_csv = pd.read_csv(file_path, sep=',')
    #     columns = list(captions_csv.columns)
    #     if columns[0] == 'Sr No':
    #         captions_csv.drop(columns=['Sr No'], inplace=True)
    #     columns = list(captions_csv.columns)
    #     for index, row in captions_csv.iterrows():
    #         caption = row[columns[1]]
    #         if caption is np.nan:
    #             continue
    #         elif all([char in caption for char in non_alpha_num]):
    #             for char in non_alpha_num:
    #                 if char in caption:
    #                     caption.replace(char, "")
    #         df['image'].append(row[columns[0]])
    #         caption = addCaptionTags(caption)
    #         df['caption'].append(caption)
    df = pd.read_csv(file_paths[0], sep=',', encoding='utf-8', header=0,index_col=0)
    df.dropna(axis=0, how='any', inplace=True)
    df['Caption'] = addCaptionTags(df['Caption'])
    return df
