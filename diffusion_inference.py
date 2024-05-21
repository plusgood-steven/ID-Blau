import argparse
import torch
import torch.nn as nn
import os
import logging
from torchvision.utils import save_image
import tqdm
from torch.utils.data import DataLoader
from itertools import islice
import random
import cv2
import sys
import numpy as np
from dataloader import Flow_Loader
from models.diffusion_model import UNet
from models.diffusion_network import DDIM, DDPM
from utils.set_condition import select_condition_strategy
from utils.flow_viz import flow_to_image
from utils.utils import same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
import pyiqa
import time
import datetime

@torch.no_grad()
def valid(model, dataloader_val, sample_timesteps, device, valid_iters=None, title=None):
    model.eval()
    psnr_func = pyiqa.create_metric('psnr', device=device)
    lpips_func = pyiqa.create_metric('lpips', device=device)
    niqe_func = pyiqa.create_metric('niqe', device=device)
    total_val_psnr = AverageMeter()
    total_val_lpips = AverageMeter()
    total_val_niqe = AverageMeter()
    if valid_iters:
        tq = tqdm.tqdm(islice(dataloader_val, valid_iters), total=valid_iters)
    else:
        tq = tqdm.tqdm(dataloader_val, total=len(dataloader_val))
    tq.set_description(f'Validation')
    start_time = time.time()
    for idx, sample in enumerate(tq):
        blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
        flow = sample['flow'].to(device)
        condition = torch.cat([sharp, flow], dim=1)
        if args.model == "DDIM":
                output = model.sample(condition=condition, sample_timesteps=sample_timesteps, device=device, tqdm_visible=False)
        elif args.model == "DDPM":
            output = model.sample(condition=condition, device=device, tqdm_visible=True)
        output = output.clamp(-0.5, 0.5)
        psnr = torch.mean(psnr_func(output.detach(), blur.detach())).item()
        lpips = torch.mean(lpips_func(output.detach(), blur.detach())).item()
        niqe = torch.mean(niqe_func(output.detach())).item()
        total_val_psnr.update(psnr)
        total_val_lpips.update(lpips)
        total_val_niqe.update(niqe)
        tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg, NIQE=total_val_niqe.avg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_obj = datetime.timedelta(seconds=elapsed_time)
    time_str = str(time_obj).split(".")[0]
    logging.info(f"-----------EVAL------------")
    logging.info(f"Title : {title}")
    logging.info(f"sample_timesteps : {sample_timesteps}")
    logging.info(f"The program's running time is (h:m:s) : {time_str}")
    logging.info(f"PSNR : {total_val_psnr.avg:.4f}, LPIPS : {total_val_lpips.avg:.4f}, NIQE : {total_val_niqe.avg:.4f}")

def val_save_image(model, dir_path, dataset, sample_timesteps, val_num=3, val_idxs=None):
    """use dataset to val and save image"""
    dir_path = os.path.join(dir_path, "images")
    os.makedirs(dir_path, exist_ok=True)
    with torch.no_grad():
        model.eval()
        if val_idxs is None:
            val_idxs = random.sample(range(0, len(dataset)), val_num)
        for i, idx in  enumerate(val_idxs):
            print(i)
            sample = dataset[idx]
            save_sharp_path = os.path.join(dir_path, 'sharp')
            os.makedirs(save_sharp_path, exist_ok=True)
            save_sharp_image_path = os.path.join(save_sharp_path, f'{idx:05d}.png')
            save_image(sample['sharp'].squeeze(0).cpu() + 0.5, save_sharp_image_path)

            save_blur_path = os.path.join(dir_path, 'blur')
            os.makedirs(save_blur_path, exist_ok=True)
            save_blur_image_path = os.path.join(save_blur_path, f'{idx:05d}.png')
            save_image(sample['blur'].squeeze(0).cpu() + 0.5, save_blur_image_path)

            sharp = sample['sharp'].unsqueeze(0).to(device)
            flow = sample['flow'].unsqueeze(0).to(device)
            condition = torch.cat([sharp, flow], dim=1)
            if args.model == "DDIM":
                output = model.sample(condition=condition, sample_timesteps=sample_timesteps, device=device, tqdm_visible=True)
            elif args.model == "DDPM":
                output = model.sample(condition=condition, device=device, tqdm_visible=True)
            output = output.clamp(-0.5, 0.5)

            save_dir_path = os.path.join(dir_path, f'output')
            os.makedirs(save_dir_path, exist_ok=True)
            save_img_path = os.path.join(save_dir_path, f'{idx:05d}.png')
            output = tensor2cv(output + 0.5)
            
            cv2.imwrite(save_img_path, output)

            #---------flow--------------------
            flow = flow.squeeze(0).cpu().numpy()
            flow = flow.transpose((1,2,0))
            flow_x = flow[:, :, 0] * flow[:, :, 2]
            flow_y = flow[:, :, 1] * flow[:, :, 2]
            optical_flow = np.stack((flow_x, flow_y), axis=-1)

            flo = flow_to_image(optical_flow, norm=1)

            flow_dir_path = os.path.join(dir_path, f'flow')
            os.makedirs(flow_dir_path, exist_ok=True)
            flow_img_path = os.path.join(flow_dir_path, f'{idx:05d}.png')
            cv2.imwrite(flow_img_path, flo[:, :, [2,1,0]])

def generate_dataset(model, dir_path, dataset, sample_timesteps, strategySetting, generate_num=5, save_npy=False):
    """use dataset to generate different image"""
    sharp_path = os.path.join(dir_path, "sharp")
    blur_path = os.path.join(dir_path, "blur")
    condition_path = os.path.join(dir_path, "condition")
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(sharp_path)
    os.makedirs(blur_path)
    os.makedirs(condition_path)

    if 'TURN' not in strategySetting:
        strategy = strategySetting[:]
    else:
        strategy_list = strategySetting[:]
        strategy_list.remove('TURN')
        if 'FIXED' in strategySetting:
            strategy_list.remove("FIXED")

    with torch.no_grad():
        model.eval()
        dataset_len = len(dataset)
        tq = tqdm.tqdm(range(dataset_len))
        tq.set_description(f'Generate images')
        for idx in tq:
            sample = dataset[idx]
            sharp_idx_path = os.path.join(sharp_path, f"{idx:05d}")
            os.makedirs(sharp_idx_path)
            save_sharp_image_path = os.path.join(sharp_idx_path, f'sharp.png')
            save_image(sample['sharp'].squeeze(0).cpu() + 0.5, save_sharp_image_path)

            blur_idx_path = os.path.join(blur_path, f"{idx:05d}")
            os.makedirs(blur_idx_path)

            if save_npy:
                condition_idx_path = os.path.join(condition_path, f"{idx:05d}")
                os.makedirs(condition_idx_path)
            
            change_base = 0
            if 'FIXED' in strategySetting:
                change_base = random.randint(0, 100)
            for index in range(generate_num):
                sharp = sample['sharp'].unsqueeze(0).to(device)
                flow = sample['flow'].clone().unsqueeze(0).to(device)
                choice_num = None
                if 'FIXED' in strategySetting:
                    choice_num = index
                if 'TURN' in strategySetting:
                    strategy = [strategy_list[(idx + index) % len(strategy_list)]]
                new_flow = select_condition_strategy(flow, strategy=strategy, choice_num=choice_num, change_base=change_base)#, strategy
                condition = torch.cat([sharp, new_flow], dim=1)
                output = model.sample(condition=condition, sample_timesteps=sample_timesteps, device=device)
                output = output.clamp(-0.5, 0.5)  # [B, C, H, W]

                save_img_path = os.path.join(blur_idx_path, f'{index:05d}.png')
                output = tensor2cv(output + 0.5)
            
                cv2.imwrite(save_img_path, output)
                
                if save_npy:
                    condition_np = new_flow.squeeze(0).cpu().numpy()
                    save_npy_path = os.path.join(condition_idx_path, f'{index:05d}.npy')
                    np.save(save_npy_path, condition_np)


def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/GOPRO_Large', type=str)
    parser.add_argument("--dir_path", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--flow_data_path",default="./dataset/GOPRO_flow",type=str)
    parser.add_argument("--flow_norm",default=True,type=bool)
    parser.add_argument("--model", default='DDIM', type=str)
    parser.add_argument("--title", default='None', type=str)
    parser.add_argument("--type", default='generate_dataset', type=str, choices=['generate_dataset', 'image'] + pyiqa.list_models())
    parser.add_argument("--dataset", default='train', type=str, choices=['train', 'test'])
    parser.add_argument("--val_num", default=5, type=int)
    parser.add_argument("--strategy", default=[], type=str, choices=['O', 'M10', 'M20', 'M30', 'M40', 'M60', 'M80', 'ALLM', 'ALLO', 'RO', '30O', '60O', 'FIXED', 'TURN'], nargs='+')
    parser.add_argument("--sample_timesteps", default=20, type=int)
    parser.add_argument("--generate_num", default=5, type=int)
    parser.add_argument("--valid_iters", default=None, type=int)
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--save_npy", default=False, type=bool)
    parser.add_argument("--seed", default=2023, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    same_seed(args.seed)
    
    load_model_state = torch.load(args.model_path)
    model_args = load_model_state['args']

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    # type: list(number) -> Specify the indices of the images or videos to be generated.
    # type: None -> Generate randomly based on the specified number, according to the value provided in args.val_num.
    val_idxs = None

    # dataset
    if args.dataset == "train":
        dataset = Flow_Loader(data_path=args.data_path,
                            flow_path=args.flow_data_path,
                            mode="train",
                            crop_size=args.crop_size,
                            flow_norm=args.flow_norm)
    elif args.dataset == "test":
        dataset = Flow_Loader(data_path=args.data_path,
                            flow_path=args.flow_data_path,
                            mode="test",
                            crop_size=args.crop_size,
                            flow_norm=args.flow_norm)
    else:
        raise ValueError("Invalid dataset type (only train and test)")
    
    # search_str = [ "GOPR0384_11_04/blur/002828.png"] 

    # val_idxs = [i for i, item in enumerate(dataset.blur_list) if any(substr in item for substr in search_str)]


    beta = generate_linear_schedule(
        model_args.num_timesteps, model_args.beta_1, model_args.beta_T)
    model_UNet = UNet(
        channel_mults=model_args.channel_mults,
        base_channels=model_args.base_channels,
        time_dim=model_args.time_dim,
        dropout=model_args.dropout
        ).to(device)
    
    if args.model == "DDIM":
        diffusionModel = DDIM(model_UNet, img_channels=9, betas=beta).to(device)
    elif args.model == "DDPM":
        diffusionModel = DDPM(model_UNet, img_channels=9, betas=beta).to(device)
    else:
        raise ValueError(f"model not supported {args.model}")
    
    if 'model_state' in load_model_state.keys():
        diffusionModel.load_state_dict(load_model_state["model_state"])
    else:
        diffusionModel.load_state_dict(load_model_state)

    print("device:", device)
    print(f'args: {args}')
    #print(f'model: {diffusionModel}')
    print(f'model parameters: {count_parameters(diffusionModel)}')

    if args.type == 'generate_dataset':
        print(f'strategy: {args.strategy}')
        generate_dataset(diffusionModel, args.dir_path, dataset, sample_timesteps=args.sample_timesteps, generate_num=args.generate_num, strategySetting=args.strategy, save_npy=args.save_npy)
    if args.type in pyiqa.list_models():
        logging.basicConfig(
        filename=os.path.join(args.dir_path, 'eval.log') , format='%(asctime)s | %(levelname)s : %(message)s', encoding='utf-8', level=logging.INFO)
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                            drop_last=False)
        
        valid(diffusionModel, dataloader, sample_timesteps=args.sample_timesteps, device=device, valid_iters=args.valid_iters, title=args.title)
    elif args.type == "image":
        val_save_image(diffusionModel, args.dir_path, dataset, sample_timesteps=args.sample_timesteps, val_num=args.val_num, val_idxs=val_idxs)
