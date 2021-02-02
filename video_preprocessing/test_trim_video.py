import subprocess
import os
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def process_vid():

  path_to_videos = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\videos\\'
  videos = os.listdir(path_to_videos)

  for video in videos:
    duration = mp.VideoFileClip(path_to_videos + video).duration
    ffmpeg_extract_subclip(path_to_videos + video, 60.0, duration - 60.0,
                           targetname=".\\output_vid\\" + video[:-4] + ".mp4")

def gen_csv():

  path_to_videos = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\output_vid\\'
  videos = os.listdir(path_to_videos)
  path_to_csv = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\scenes\\'

  for video in videos:
    x = subprocess.run(["scenedetect","--input" ,path_to_videos+video, "detect-content", "list-scenes", '--output', path_to_csv],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(x)

if __name__ == '__main__':
    #process_vid()
    gen_csv()

