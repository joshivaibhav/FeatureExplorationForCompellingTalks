import numpy as np
import pandas as pd
import os

def point_pair_distances(joint_1, joint_2, points_data):
    # get the point data for these joints....
    p0 = points_data[:, joint_1 * 2:joint_1 * 2 + 2]
    p1 = points_data[:, joint_2 * 2:joint_2 * 2 + 2]

    # calculate the pair-wise distance and difference
    diff = p0 - p1
    diff_square = diff ** 2
    diff_sum_col = np.sum(diff_square, axis=1)
    pair_dist = np.sqrt(diff_sum_col)

    # check the special case when point is not captured(0, 0, 0)
    sum_p0 = p0.sum(axis=1)
    sum_p1 = p1.sum(axis=1)
    min_sum = np.minimum(sum_p0, sum_p1)

    # keep the valid information only(without special cases)
    pair_dist = pair_dist[min_sum > 0.0]
    diff = diff[min_sum > 0.0]

    return pair_dist, diff


def time_seg_avg_dist(dist, seg_len, ignore_value=-10):

    frames = np.arange(dist.shape[0])
    avg_frames = []
    avg_dists = []
    var_dists = []
    for ind in range(0, dist.shape[0], seg_len):
        if ind + seg_len - 1 >= dist.shape[0]:
            break

        # middle position of each small sequencial segment
        avg_frames.append((frames[ind] + frames[ind + seg_len - 1]) / 2)
        segment_dist = dist[ind:ind + seg_len]

        # check special case when one point in the pair is not detected: (0, 0, 0)
        segment_dist = [item for item in segment_dist if item != ignore_value]
        if segment_dist == []:
            avg_dists.append(ignore_value)
            var_dists.append(ignore_value)
            continue

        avg_dist = np.mean(segment_dist)
        var_dist = np.var(segment_dist)
        avg_dists.append(avg_dist)
        var_dists.append(var_dist)

    return avg_dists

def test_normalize():

    body_segments = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_segments_all')
    body_segments_normalized = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_segments_normalized_new')

    for file in os.listdir(body_segments):

        df = pd.read_csv(body_segments+ "\\" +file)
        col = df.columns[0]
        if "Unnamed" in col:
            df.drop(df.columns[0],axis=1,inplace=True)
            pair_dist,_ = point_pair_distances(1,8,df.values)
            norm_factor = time_seg_avg_dist(pair_dist, 30)

            df['Norm_Factor'] = pd.Series([norm_factor[0]] * len(df))
            print(len(df.columns))
            df.to_csv(body_segments_normalized + "\\" + file, index=False)
        else:
            pair_dist, _ = point_pair_distances(1, 8, df.values)
            norm_factor = time_seg_avg_dist(pair_dist, 30)

            df['Norm_Factor'] = pd.Series([norm_factor[0]] * len(df))
            print(len(df.columns))
            df.to_csv(body_segments_normalized + "\\" + file, index=False)

        """
        for x_col in df.columns[:37:2]:
            for idx,val in enumerate(df[x_col]):
                if df['pose_x1'][idx] - df['pose_x0'][idx] == 0.0 or x_col == 'pose_x0' or x_col == 'pose_x1':
                    continue
                else:
                    df[x_col][idx] = abs(val - df['pose_x1'][idx]) / abs(df['pose_x1'][idx] - df['pose_x0'][idx])

        for y_col in df.columns[1:38:2]:
            for idx,val in enumerate(df[y_col]):
                if df['pose_y1'][idx] - df['pose_y0'][idx] == 0.0 or y_col == 'pose_y0' or y_col == 'pose_y1':
                    continue
                else:
                    df[y_col][idx] = abs(val - df['pose_y1'][idx]) / abs(df['pose_y1'][idx] - df['pose_y0'][idx])

        df = df.iloc[:,:38]
        cols = df.columns
        df = MinMaxScaler().fit_transform(df.values)

        """



if __name__ == '__main__':
    test_normalize()