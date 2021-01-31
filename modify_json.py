import json
import os


def label_json():

    json_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agcn_json_files')

    for json_file in os.listdir(json_directory):
        with open(json_directory+"//"+json_file, "r") as read_file:
            data = json.load(read_file)
        if data["label"] == "low likes":
            data["label_index"] = 0
        elif data["label"] == "moderate likes":
            data["label_index"] = 1
        else:
            data["label_index"] = 2

        with open(json_directory + "//" + json_file, "w") as fp:
            json.dump(data, fp, indent=2)

def features_json():

    label = 'train_label.json'

    with open(label, "r") as read_file:
        data = json.load(read_file)

    for key in data:
        if data[key]["label"] == "low likes":
            data[key]["label_index"] = 0
        elif data[key]["label"] == "moderate likes":
            data[key]["label_index"] = 1
        else:
            data[key]["label_index"] = 2

    with open(label, "w") as fp:
        json.dump(data, fp, indent=2)


if __name__ == '__main__':

    label_json()
    features_json()