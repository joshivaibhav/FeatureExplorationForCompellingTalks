import pandas as pd
import numpy as np
import os

def remove_header():

    path_to_scenes = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\scenes\\'

    for scene_file in os.listdir(path_to_scenes):
        with open(path_to_scenes+scene_file,'r') as f:
            data = f.read().splitlines(True)
        with open(path_to_scenes+scene_file, 'w') as fout:
                fout.writelines(data[1:])


if __name__ == '__main__':
    remove_header()
