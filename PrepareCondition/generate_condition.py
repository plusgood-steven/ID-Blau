import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import json
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

def load_image(imfile, device='cuda'):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def generate(args, comparative_dict, mode, device):
    origin_video_list = os.listdir(os.path.join(args.origin_dataset_path, mode))
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(device)
    model.eval()
    
    magnitude_max=torch.tensor([0]).cuda()

    for video in origin_video_list:
        composite_video_path = os.path.join(args.composite_dataset_path, mode, video, "blur")
        origin_video_path = os.path.join(args.origin_dataset_path, mode, video)
        composite_img_name_list = sorted(os.listdir(composite_video_path), key=lambda name: int(name[:-4]))
        origin_img_name_list = sorted(os.listdir(origin_video_path), key=lambda name: int(name[:-4]))

        flow_video_path = os.path.join(args.dir_path, mode, video)
        os.makedirs(flow_video_path, exist_ok=True)

        #Read all image data from a video
        pbar = tqdm(composite_img_name_list)
        pbar.set_description(f'Reading composite {video}')
        for img_name in pbar:
            origin_img_name = comparative_dict[video][img_name]["origin_img_name"]
            origin_img_frames_num = comparative_dict[video][img_name]["frames_num"]
            origin_img_name_index = origin_img_name_list.index(origin_img_name)

            #composite_image_path = os.path.join(composite_video_path, img_name)
            #blurry_image =load_image(composite_image_path)

            if origin_img_frames_num == 0:
                continue

            first_index = origin_img_name_index - (origin_img_frames_num // 2)
            origin_images_select_name_list = origin_img_name_list[first_index:(first_index + origin_img_frames_num)]
            assert len(origin_images_select_name_list) == origin_img_frames_num, "must be same length"
            with torch.no_grad():
                flow0 = 0
                flow1 = 0
                for j in range(origin_img_frames_num-1):
                    origin_image1_path = os.path.join(origin_video_path, origin_images_select_name_list[j])
                    origin_imagen_path = os.path.join(origin_video_path, origin_images_select_name_list[j + 1])
                    image1 = load_image(origin_image1_path)
                    imagen = load_image(origin_imagen_path)
            
                    padder = InputPadder(image1.shape)
                    image1, imagen = padder.pad(image1, imagen)
            
                    flow_low_1n, flow_up_1n = model(image1, imagen, iters=args.epochs, test_mode=True)
                    
                    flow0 += flow_up_1n
                
                for j in reversed(range(1 ,origin_img_frames_num)):
                    origin_image1_path = os.path.join(origin_video_path, origin_images_select_name_list[j])
                    origin_imagen_path = os.path.join(origin_video_path, origin_images_select_name_list[j - 1])
                    image1 = load_image(origin_image1_path)
                    imagen = load_image(origin_imagen_path)
            
                    padder = InputPadder(image1.shape)
                    image1, imagen = padder.pad(image1, imagen)
            
                    flow_low_1n, flow_up_1n = model(image1, imagen, iters=args.epochs, test_mode=True)
                    
                    flow1 -= flow_up_1n

                flow0 = flow0.cpu().squeeze().numpy()
                flow1 = flow1.cpu().squeeze().numpy()
                flow = (flow0 + flow1) / 2

                magnitude = np.linalg.norm(flow, axis=0)
                magnitude = np.expand_dims(magnitude, axis=0)

                flow = flow / magnitude #transfer to unit vector
                info = np.concatenate((flow, magnitude), axis=0).astype(np.float32)

                save_name = img_name.replace("png","npy")
                save_path = os.path.join(flow_video_path, save_name)
                np.save(save_path, info)
                
                if np.max(magnitude) > magnitude_max:
                    magnitude_max=np.max(magnitude)

                #viz(blurry_image, temp)

            
    print("magnitude_max", magnitude_max)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dir_path', help="save dataset path")
    parser.add_argument('--mode', default='all', help="all or train or test")
    parser.add_argument('--json_dir_path', default='./detail', help="json dataset path")
    parser.add_argument('--origin_dataset_path', default="../dataset/GOPRO_Large_all", help="dataset path")
    parser.add_argument('--composite_dataset_path',default="../dataset/GOPRO_Large", help="dataset path")
    parser.add_argument('--epochs',default=20, help="iter times")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    modes = []
    if args.mode == 'all':
        modes = ['train', 'test']
    
    for mode in modes:
        tf = open(os.path.join(args.json_dir_path, f"{mode}_composite_img_frames.json"), "r")
        new_dict = json.load(tf)
        generate(args, new_dict, mode, device)