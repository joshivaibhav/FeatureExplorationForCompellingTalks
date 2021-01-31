import pandas as pd
import numpy as np
import os
from pprint import pprint


def preproces_csvs():

    flag = True
    path_to_csvs = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\scenes'
    for file in os.listdir(path_to_csvs):
        with open(path_to_csvs+"\\"+file, 'r') as fin:
            data = fin.read().splitlines(True)
            if "Start" in data[0]:
                flag = False
            else:
                flag = True
        if flag:
            with open(path_to_csvs+"\\"+file, 'w') as fout:
                fout.writelines(data[1:])
                fout.close()


def isbodysegment(segment):

    if (np.count_nonzero(segment) / (len(segment) * 50)) > 0.20:
        return True
    return False


def isfacesegment(segment,filename):

    return False


def prepare_body_csv(body_segments, files):

    body_csv_path = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\body_segments_all\\'
    bodycsv = pd.concat(body_segments,ignore_index=True)
    bodycsv.to_csv(body_csv_path+"body_"+files, index=False)



def scene_mapping():

    scene_mappings = {}

    path_to_scenes = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\scenes\\'
    path_to_csvs = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\processedcsvs\\'

    for csv in os.listdir(path_to_scenes):
        file = pd.read_csv(path_to_scenes+"\\"+csv)
        scenes = []
        for start,end in zip(file['Start Frame'],file['End Frame']):
            scenes.append((start,end))
        scene_mappings[csv[:-11]] = scenes

    # hand_segments = {}
    # face_segments = {}

    for files in os.listdir(path_to_csvs):
        body_segments = []
        video = pd.read_csv(path_to_csvs + files)
        for mapping in scene_mappings[files[:-4]]:
                segment = video.iloc[mapping[0]:mapping[1],:]
                if isbodysegment(segment):
                    body_segments.append(segment)

        prepare_body_csv(body_segments,files)



if __name__ == '__main__':

    #preproces_csvs()
    scene_mapping()