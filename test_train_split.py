import json
import shutil
import os
import time


def split_filenames(label_file):

    with open(label_file, 'r') as fp:
        data = json.load(fp)

    all = set()
    for key in data:
        id = key.rsplit("_",1)[0]
        all.add(id)

    all = list(all)
    train,test = all[:int(len(all)*0.8)], all[int(len(all)*0.8):]
    return train, test

def split_files(json_directory,label_file):

    train,test = split_filenames(label_file)
    json_files = os.listdir(json_directory)
    for id in test:
        dir_files = [file for file in json_files if id in file]
        start_time = time.time()
        for file in dir_files:
            shutil.copy(json_directory+"\\"+file, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       'kinetics_val'))
        print(id + "--- %s seconds ---" % (time.time() - start_time))

    for id in train:
        dir_files = [file for file in json_files if id in file]
        start_time = time.time()
        for file in dir_files:
            shutil.copy(json_directory+"\\"+file, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       'kinetics_train'))
        print(id + "--- %s seconds ---" % (time.time() - start_time))

    split_label(train,test,label_file)

def split_label(train,test,label_file):

    with open(label_file,'r') as fp:
        json_data = json.load(fp)

    train_json = {}
    test_json = {}

    for id in train:
        keys = [key for key in json_data.keys() if id in key]
        for key in keys:
            train_json[key] = json_data[key]

    with open('kinetics_train_label.json', 'w+') as ftrain:
        json.dump(train_json,ftrain,indent=2)

    for id in test:
        keys = [key for key in json_data.keys() if id in key]
        for key in keys:
            test_json[key] = json_data[key]

    with open('kinetics_val_label.json', 'w+') as ftest:
        json.dump(test_json,ftest,indent=2)


if __name__ == '__main__':
    split_files("C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\agcn_json_files", "train_label1.json")
    #split_label("train_label1.json")