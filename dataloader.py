#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random

def rotation_matrix(angle):
    rad = np.radians(angle)
    
    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)
    rot_matrix = np.array([[cos_theta, -sin_theta],
                           [sin_theta, cos_theta]])
    
    return rot_matrix

class RandomRotate(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            if key != 'flow':
                data[key] = np.rot90(data[key], dirct).copy()
            else:
                vectors = data[key][:, : ,:2].copy()
                
                vectors_origin_shape = vectors.shape
                vectors = vectors.reshape((-1, 2))
                rot_matrix = rotation_matrix(90 * dirct)
                
                rotated_vectors = (rot_matrix@vectors.T).T
                rotated_vectors = rotated_vectors.reshape(vectors_origin_shape)
                
                data[key][:, :, :2] = rotated_vectors
                
        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != 'flow':
                    data[key] = np.fliplr(data[key]).copy()
                else:
                    data[key][:, :, 0] = -data[key][:, :, 0]

        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != 'flow':
                    data[key] = np.flipud(data[key]).copy()
                else:
                    data[key][:, :, 1] = -data[key][:, :, 1]   
        return data

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data

class Normalize(object):
    def __init__(self, ZeroToOne=False):
        super(Normalize, self).__init__()
        self.ZeroToOne = ZeroToOne
        self.num = 0 if ZeroToOne else 0.5

    def __call__(self, data):
        for key in data.keys():
            if key != 'flow':
                data[key] = ((data[key] / 255) - self.num).copy()
        return data

class ToTensor(object):
    def __call__(self, data):
        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()
        return data

class Flow_Loader(Dataset):
    def __init__(self, data_path, flow_path, mode, crop_size=None, flow_norm=True):
        self.blur_list = []
        self.sharp_list = []
        self.flow_list = []
        self.flow_norm = flow_norm
        self.flow_norm_num = 147
        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(), ToTensor()])

        for video in sorted(os.listdir(os.path.join(data_path, mode))):
            flow_video_path = os.path.join(flow_path, mode, video)
            data_blur_video_path = os.path.join(data_path, mode, video, 'blur')
            data_sharp_video_path = os.path.join(data_path, mode, video, 'sharp')
            flow_video_data_path = sorted(glob.glob(os.path.join(flow_video_path, '*.npy')))
            self.flow_list.extend(flow_video_data_path)
            self.blur_list.extend([ npy_path.replace(flow_video_path, data_blur_video_path).replace('.npy', '.png') for npy_path in flow_video_data_path])
            self.sharp_list.extend([ npy_path.replace(flow_video_path, data_sharp_video_path).replace('.npy', '.png') for npy_path in flow_video_data_path])

        assert len(self.flow_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.flow_list)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_list[idx]).astype(np.float32)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        flow = np.load(self.flow_list[idx])

        if self.flow_norm:
            magnitude = flow[2] / self.flow_norm_num
            magnitude[magnitude > 1] = 1
            flow[2] = magnitude
        flow = flow.transpose((1, 2, 0))

        sample = {'blur': blur,
                  'sharp': sharp,
                  'flow': flow}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_path(self, idx):
        return {'flow_path': self.flow_list[idx]}
    
class Multi_GoPro_Loader(Dataset):
    def __init__(self, data_path=None, generate_path=None, mode="train", crop_size=None, ZeroToOne=False, video_generate_path=None):
        """generate_path can be str or list"""
        assert data_path or generate_path or video_generate_path, "must have one dataset path !"
        self.blur_list = []
        self.sharp_list = []

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])

        if data_path:
            for video in sorted(os.listdir(os.path.join(data_path, mode))):
                self.blur_list.extend(sorted(glob.glob(os.path.join(data_path, mode, video, "blur", '*.png'))))
                self.sharp_list.extend(sorted(glob.glob(os.path.join(data_path, mode, video, "sharp", '*.png'))))

        if generate_path and mode == "train":
            if isinstance(generate_path, str):
                generate_path = [generate_path]
            
            for now_generate_path in generate_path:
                sharp_image_folders_list =  sorted(os.listdir(os.path.join(now_generate_path, "sharp")))
                for folder in sharp_image_folders_list:
                    blur_images_list = sorted(glob.glob(os.path.join(now_generate_path, "blur", folder , '*.png')))
                    self.blur_list.extend(blur_images_list)
                    
                    blur_length = len(blur_images_list)
                    sharp_images_list = (glob.glob(os.path.join(now_generate_path, "sharp", folder, 'sharp.png'))) * blur_length
                    self.sharp_list.extend(sharp_images_list)

        if video_generate_path and mode == "train":
            if isinstance(video_generate_path, str):
                video_generate_path = [video_generate_path]
            for now_generate_path in video_generate_path:
                video_image_folders_list =  sorted(os.listdir(os.path.join(now_generate_path, "sharp")))
                for video in video_image_folders_list:
                    sharp_image_folders_list = sorted(os.listdir(os.path.join(now_generate_path, "sharp", video)))
                    for folder in sharp_image_folders_list:
                        blur_images_list = sorted(glob.glob(os.path.join(now_generate_path, "blur", video, folder , '*.png')))
                        self.blur_list.extend(blur_images_list)
                        
                        blur_length = len(blur_images_list)
                        sharp_images_list = (glob.glob(os.path.join(now_generate_path, "sharp", video, folder, 'sharp.png'))) * blur_length
                        self.sharp_list.extend(sharp_images_list)
        
        assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.sharp_list)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_list[idx]).astype(np.float32)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        sample = {'blur': blur,
                  'sharp': sharp}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RealBlur_Loader(Dataset):
    def __init__(self, data_path=None, mode="train", crop_size=None, ZeroToOne=False):
        assert data_path, "must have one dataset path !"
        self.blur_list = []
        self.sharp_list = []

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])

        for video in sorted(os.listdir(os.path.join(data_path, mode, "blur"))):
            self.blur_list.extend(sorted(glob.glob(os.path.join(data_path, mode, "blur", video, '*.png'))))
            self.sharp_list.extend(sorted(glob.glob(os.path.join(data_path, mode, "sharp", video, '*.png'))))
        
        assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.sharp_list)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_list[idx]).astype(np.float32)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        sample = {'blur': blur,
                  'sharp': sharp}

        if self.transform:
            sample = self.transform(sample)

        return sample

class GoPro_RealBlur_Loader(Dataset):
    def __init__(self, GoPro_data_path=None, Realblur_data_path=None, mode="train", crop_size=None, ZeroToOne=False):
        """generate_path can be str or list"""
        assert GoPro_data_path or Realblur_data_path, "must have one dataset path !"
        self.blur_list = []
        self.sharp_list = []

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])

        for video in sorted(os.listdir(os.path.join(GoPro_data_path, mode))):
            self.blur_list.extend(sorted(glob.glob(os.path.join(GoPro_data_path, mode, video, "blur", '*.png'))))
            self.sharp_list.extend(sorted(glob.glob(os.path.join(GoPro_data_path, mode, video, "sharp", '*.png'))))

        for video in sorted(os.listdir(os.path.join(Realblur_data_path, mode, "blur"))):
            self.blur_list.extend(sorted(glob.glob(os.path.join(Realblur_data_path, mode, "blur", video, '*.png'))))
            self.sharp_list.extend(sorted(glob.glob(os.path.join(Realblur_data_path, mode, "sharp", video, '*.png'))))
        
        assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.sharp_list)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_list[idx]).astype(np.float32)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        sample = {'blur': blur,
                  'sharp': sharp}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Test_Loader(Dataset):
    def __init__(self, data_path=None, crop_size=None, ZeroToOne=False):
        assert data_path , "must have one dataset path !"
        self.blur_list = []
        self.sharp_list = []
        self.is_sharp_dir = os.path.isdir(os.path.join(data_path, "target"))

        if crop_size:
            self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), Normalize(ZeroToOne), ToTensor()])
        else:
            self.transform = transforms.Compose([Normalize(ZeroToOne), ToTensor()])

        if data_path:
            self.blur_list.extend(sorted(glob.glob(os.path.join(data_path, "input", '*.png'))))
            if self.is_sharp_dir:
                self.sharp_list.extend(sorted(glob.glob(os.path.join(data_path, "target", '*.png'))))
        
        if self.is_sharp_dir:
            assert len(self.sharp_list) == len(self.blur_list), "Missmatched Length!"

    def __len__(self):
        return len(self.blur_list)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_list[idx]).astype(np.float32)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        if self.is_sharp_dir:
            sharp = cv2.imread(self.sharp_list[idx]).astype(np.float32)
            sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

            sample = {'blur': blur,
                    'sharp': sharp}
        else:
            sample = {'blur': blur}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_path(self, idx):
        if self.is_sharp_dir:
            return {'blur_path': self.blur_list[idx], 'sharp_path': self.sharp_list[idx]}
        else:
            return {'blur_path': self.blur_list[idx]}

def get_image(path):
    transform = transforms.Compose([Normalize(), ToTensor()])
    image = cv2.imread(path).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = {'image': image}
    sample = transform(sample)

    return sample['image']

if __name__ == "__main__":
    dataloader = Flow_Loader(
        data_path='./dataset/GOPRO_Large',
        flow_path= './dataset/GOPRO_flow',
        mode="train",
        crop_size=128,
        flow_norm=True
        )
    
    print(dataloader.sharp_list[-10:])
    print(dataloader.blur_list[-10:])
    print(len(dataloader))
#%%

