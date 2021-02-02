import json
import pandas as pd
import os


def create_json(data_file):

    #data = pd.read_csv(data_file)
    json_head = {}
    json_keypoint_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agcn_json_files')

    for file in os.listdir(json_keypoint_files):

        file_name = file.split(".")[0]
        with open(json_keypoint_files + "\\" + file , 'r') as fp:
            segment_data = json.load(fp)

        label = segment_data["label"]
        label_index = segment_data["label_index"]
        #print(segment_id+"_"+segment_no)
        json_head[file_name] = {
            "has_skeleton": True,
            "label": label,
            "label_index": label_index
        }

    with open('train_label1.json', 'w+') as fp:
        json.dump(json_head, fp, indent=2)

if __name__ == '__main__':

    create_json('C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\metadata.csv')
