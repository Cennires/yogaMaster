import argparse
import glob
import json
import numpy as np
import os

from .pose import Pose, Part, PoseSequence
from pprint import pprint


# def main():
#
#     parser = argparse.ArgumentParser(description='Pose Trainer Parser')
#     parser.add_argument('--input_folder', type=str, default='poses', help='input folder for json files')
#     parser.add_argument('--output_folder', type=str, default='poses_compressed', help='output folder for npy files')
#
#     args = parser.parse_args()
#     # poses中有很多输入文件
#     video_paths = glob.glob(os.path.join(args.input_folder, '*'))
#     video_paths = sorted(video_paths)
#
#     # Get all the json sequences for each video
#     # video_paths：输入视频路径；
#     # all_ps：输入视频路径+对应关键点输出文件路径；
#     all_ps = []
#     for video_path in video_paths:
#         all_ps.append(parse_sequence(video_path, args.output_folder))
#     return video_paths, all_ps


# def parse_sequence(json_folder, output_folder):
def parse_sequence(path, imgid):
    """Parse a sequence of OpenPose JSON frames and saves a corresponding numpy file.

    Args:
        json_folder: path to the folder containing OpenPose JSON for one video.
        output_folder: path to save the numpy array files of keypoints.

    """
    json_files = [path]
    json_files = sorted(json_files)
    # 总共有num_frames个输入文件，输入文件为json文件
    num_frames = len(json_files)
    all_keypoints = np.zeros((num_frames, 25, 3))
    for i in range(num_frames):
        with open(json_files[i]) as f:
            json_obj = json.load(f)
            keypoints = np.array(json_obj['people'][0]['pose_keypoints_2d'])
            # all_keypoints[i] = keypoints.reshape((18, 3))
            all_keypoints[i] = keypoints.reshape((25, 3))

    # output_dir = os.path.join(output_folder, os.path.basename(json_folder))
    path2 = os.path.join(os.getcwd(), 'json_npy', imgid)
    np.save(path2, all_keypoints)


def load_ps(filename):
    """Load a PoseSequence object from a given numpy file.

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    # 打开npy文件，保存关键点信息，进行PoseSequence处理
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)


# if __name__ == '__main__':
    # main()
