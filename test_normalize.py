import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def test_normalize():

    body_segments = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_segments_all')
    body_segments_normalized = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_segments_normalized')

    for file in os.listdir(body_segments):

        df = pd.read_csv(body_segments+ "\\" +file)

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

        pd.DataFrame(df,columns=cols).to_csv(body_segments_normalized+"\\"+file,index=False)

if __name__ == '__main__':
    test_normalize()