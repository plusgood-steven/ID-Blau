import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm

class FindComposite():
    def __init__(self, composite_dataset_path, origin_dataset_path):
        self.composite_dataset_path = composite_dataset_path
        self.origin_dataset_path = origin_dataset_path
    
    def find_origin_img_name(self, dir_path, mode):
        assert mode == "train" or mode == "test"
        composite_video_list = os.listdir(os.path.join(self.composite_dataset_path, mode))
        origin_video_list = os.listdir(os.path.join(self.origin_dataset_path, mode))

        json_path = os.path.join(dir_path, f"{mode}_sharp_img_origin_path.json")
        tf = open(json_path, "w")
        total_info = {}

        for video in composite_video_list:
            assert video in origin_video_list, "Video must in origin dataset"
            video_info = {}

            composite_video_path = os.path.join(self.composite_dataset_path, mode, video, "sharp")
            origin_video_path = os.path.join(self.origin_dataset_path, mode, video)
            composite_img_name_list = sorted(os.listdir(composite_video_path), key=lambda name: int(name[:-4]))
            origin_img_name_list = sorted(os.listdir(origin_video_path), key=lambda name: int(name[:-4]))

            #Read all image data from a video
            composite_img_dict = {}
            pbar = tqdm(composite_img_name_list)
            pbar.set_description(f'Reading composite {video}')
            for img_name in pbar:
                composite_image_path = os.path.join(composite_video_path, img_name)
                image = cv2.imread(composite_image_path).astype(np.float32)
                composite_img_dict[img_name] = image

            pbar = tqdm(origin_img_name_list)
            pbar.set_description(f'Reading origin {video}')
            for img_name in pbar:
                image = cv2.imread(os.path.join(origin_video_path, img_name)).astype(np.float32)
                for k, v in composite_img_dict.items():
                    if (v == image).all():
                        video_info[k] = {}
                        video_info[k]["origin_img_name"] = img_name
                        del composite_img_dict[k]
                        break

            assert len(composite_img_dict) == 0, f"Some composite images do not find origin img !"
            total_info[video] = video_info

        json.dump(total_info, tf, indent=4)
    
    def find_composite_img(self,dir_path, comparative_dict, mode):
        assert mode == "train" or mode == "test"
        composite_video_list = os.listdir(os.path.join(self.composite_dataset_path, mode))
        origin_video_list = os.listdir(os.path.join(self.origin_dataset_path, mode))

        json_path = os.path.join(dir_path, f"{mode}_composite_img_frames.json")
        tf = open(json_path, "w")
        total_info = {}

        for video in composite_video_list:
            assert video in origin_video_list, "Video must in origin dataset"
            video_info = {}

            composite_video_path = os.path.join(self.composite_dataset_path, mode, video, "blur")
            origin_video_path = os.path.join(self.origin_dataset_path, mode, video)
            composite_img_name_list = sorted(os.listdir(composite_video_path), key=lambda name: int(name[:-4]))
            origin_img_name_list = sorted(os.listdir(origin_video_path), key=lambda name: int(name[:-4]))

            #Read all image data from a video
            pbar = tqdm(composite_img_name_list)
            pbar.set_description(f'Reading composite {video}')
            for img_name in pbar:
                composite_image_path = os.path.join(composite_video_path, img_name)
                origin_img_name = comparative_dict[video][img_name]["origin_img_name"]
                origin_img_name_index = origin_img_name_list.index(origin_img_name)
                image = cv2.imread(composite_image_path).astype(np.float32)

                if video == "GOPR0372_07_00" or video == "GOPR0372_07_01":
                    check_frames = [7, 11,9,13,5,3,15]
                elif video == "GOPR0378_13_00":
                    check_frames = [13,11,9,7,5,3,15]
                else:
                    check_frames = [11,7,9,13,5,3,15]
                
                comparative_dict[video][img_name]["frames_num"] = 0

                for frames_num in check_frames:
                    first_index = origin_img_name_index - (frames_num // 2)
                    origin_images = []
                    if first_index + (frames_num) > len(origin_img_name_list) or first_index  < 0:
                        continue
                    for i in range(0, frames_num):
                        origin_image_path = os.path.join(origin_video_path, origin_img_name_list[first_index + i])
                        origin_image = cv2.imread(origin_image_path).astype(np.float32)
                        origin_images.append(origin_image)

                    blurry_img = np.around(sum(origin_images) / frames_num)

                    if (image == blurry_img).all():
                        comparative_dict[video][img_name]["frames_num"] = frames_num
                        break
                
        json.dump(comparative_dict, tf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="all", help="train or test or all")
    parser.add_argument('--dir_path',default="./detail", help="save dir path")
    parser.add_argument('--origin_dataset_path', default="../dataset/GOPRO_Large_all", help="dataset path")
    parser.add_argument('--composite_dataset_path',default="../dataset/GOPRO_Large", help="dataset path")
    args = parser.parse_args()

    Findcomposite = FindComposite(composite_dataset_path=args.composite_dataset_path, origin_dataset_path=args.origin_dataset_path)
    
    os.makedirs(args.dir_path)
    
    modes = [args.mode]
    if args.mode == "all":
        modes = ['train', 'test']

    for mode in modes:
        Findcomposite.find_origin_img_name(args.dir_path, mode)
        tf = open(os.path.join(args.dir_path, f"{mode}_sharp_img_origin_path.json"), "r")
        new_dict = json.load(tf)
        Findcomposite.find_composite_img(args.dir_path, new_dict, mode)
            