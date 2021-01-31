import numpy as np
import json
from numpy import dot
from numpy.linalg import norm


def similarity():

    with open("ted_000000000386_keypoints.json" ,"r") as read_file:
        data = json.load(read_file)

    pose_key_points = np.array_split(np.array(data["people"][0]["pose_keypoints_2d"]), 25)
    face_key_points = np.array_split(np.array(data["people"][0]["face_keypoints_2d"]), 70)
    hand_left_key_points = np.array_split(np.array(data["people"][0]["hand_left_keypoints_2d"]), 21)
    hand_right_key_points = np.array_split(np.array(data["people"][0]["hand_right_keypoints_2d"]), 21)

    pose_key_points_x = [key_point[0] for key_point in pose_key_points]
    pose_key_points_y = [key_point[1] for key_point in pose_key_points]

    face_key_points_x = [key_point[0] for key_point in face_key_points]
    face_key_points_y = [key_point[1] for key_point in face_key_points]

    hand_left_key_points_x = [key_point[0] for key_point in hand_left_key_points]
    hand_left_key_points_y = [key_point[1] for key_point in hand_left_key_points]

    hand_right_key_points_x = [key_point[0] for key_point in hand_right_key_points]
    hand_right_key_points_y = [key_point[1] for key_point in hand_right_key_points]

    im1_p = list(zip(pose_key_points_x,pose_key_points_y))
    im1_f = list(zip(face_key_points_x,face_key_points_y))

    with open("ted_000000000385_keypoints.json" ,"r") as read_file:
        data = json.load(read_file)

    pose_key_points = np.array_split(np.array(data["people"][0]["pose_keypoints_2d"]), 25)
    face_key_points = np.array_split(np.array(data["people"][0]["face_keypoints_2d"]), 70)
    hand_left_key_points = np.array_split(np.array(data["people"][0]["hand_left_keypoints_2d"]), 21)
    hand_right_key_points = np.array_split(np.array(data["people"][0]["hand_right_keypoints_2d"]), 21)

    pose_key_points_x = [key_point[0] for key_point in pose_key_points]
    pose_key_points_y = [key_point[1] for key_point in pose_key_points]

    face_key_points_x = [key_point[0] for key_point in face_key_points]
    face_key_points_y = [key_point[1] for key_point in face_key_points]

    hand_left_key_points_x = [key_point[0] for key_point in hand_left_key_points]
    hand_left_key_points_y = [key_point[1] for key_point in hand_left_key_points]

    hand_right_key_points_x = [key_point[0] for key_point in hand_right_key_points]
    hand_right_key_points_y = [key_point[1] for key_point in hand_right_key_points]

    im2_p = list(zip(pose_key_points_x,pose_key_points_y))
    im2_f = list(zip(face_key_points_x,face_key_points_y))

    a1 = np.asarray(im1_f).reshape(-1, 1)
    b1 = np.asarray(im2_f).reshape(-1, 1)

    a = np.squeeze(np.asarray(a1))
    b = np.squeeze(np.asarray(b1))

    cos_sim = dot(a, b) / (norm(a) * norm(b))
    print(cos_sim)
