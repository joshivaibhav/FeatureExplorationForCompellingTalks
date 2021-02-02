import numpy as np
import json
import os
import pandas as pd
import time

def get_video_details():
    with open("train_label.json", "r") as read_file:
        metadata = json.load(read_file)

    print([x for x in metadata.keys()])
    return metadata

def keypoints2csv():

    metadata = get_video_details()
    keypoint_directory =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json_keypoints')
    agcn_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agcn_json_files')

    for idx,keypoint_file in enumerate(os.listdir(keypoint_directory)):

        start_time = time.time()
        file_id = list(metadata.keys())[idx]

        n_files = len(os.listdir(keypoint_directory + "\\" + keypoint_file))
        n_frames, mod = divmod(n_files, 30)

        for segment in range(0, (n_frames * 30), 30):

            json_load = {"data": []}
            for idx, keypoint_json_file in enumerate(os.listdir(keypoint_directory + "\\" + keypoint_file)[segment : segment + 30]):

                skeleton = generate_json(keypoint_directory + "\\" + keypoint_file + "\\" + keypoint_json_file)

                frame_data = {"frame_index": idx}
                frame_data["skeleton"] = skeleton
                json_load["data"].append(frame_data)
                json_load["label"] = metadata[file_id]["label"]
                json_load["label_index"] = metadata[file_id]["label_index"]


            with open(agcn_json_path + "\\" + file_id + '_' + str(segment // 30)  + '.json', 'w+') as fp:
                json.dump(json_load, fp, indent=2)
        print(keypoint_file + "------------------",end=" ")
        print("--- %s seconds ---" % (time.time() - start_time))



def generate_json(key_point_file):
    with open(key_point_file, "r+") as read_file:
        data = json.load(read_file)

    if len(np.array(data["people"])) > 0:

        pose_key_points = np.array_split(np.array(data["people"][0]["pose_keypoints_2d"]), 25)
        pose_key_points_x = [key_point[0] for key_point in pose_key_points]
        pose_key_points_y = [key_point[1] for key_point in pose_key_points]
        pose_key_points_c = [key_point[2] for key_point in pose_key_points]

        frame_json = {"pose" : [], "score" : []}

        i = 0
        while i < 19:
            if i == 8:
                pass
            else:
                frame_json["pose"].append(pose_key_points_x[i])
                frame_json["pose"].append(pose_key_points_y[i])
                frame_json["score"].append(pose_key_points_c[i])
            i += 1
        return [frame_json]
    else:
        return []


if __name__ == '__main__':

    keypoints2csv()








