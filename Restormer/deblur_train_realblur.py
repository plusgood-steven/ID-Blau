import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import tqdm
import cv2
import os
import argparse
import logging
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import RealBlur_Loader
from Restormer.model import Restormer
from Restormer.losses import L1Loss
from utils.utils import calc_psnr, same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
import torch.nn.functional as F
import pyiqa
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Trainer():
    def __init__(self, model, optimizer, scheduler, args, writer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.device = self.args.device
        self.psnr_func = pyiqa.create_metric('psnr', device=device)
        self.lpips_func = pyiqa.create_metric('lpips', device=device)
        self.best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0
        self.grad_clip = 0.01

        self.mini_epochs = args.mini_epochs
        self.mini_batch_sizes = args.mini_batch_sizes
        self.mini_crop_sizes = args.mini_crop_sizes
        assert len(self.mini_epochs) == len(self.mini_batch_sizes) and len(self.mini_epochs) == len(self.mini_crop_sizes), "progressive setting must have same len"

        self.progressive_idx=0
        self.threshold_epochs = self.mini_epochs[self.progressive_idx]
        self.now_batch_size_per_gpu = self.mini_batch_sizes[self.progressive_idx]
        self.now_crop_size = self.mini_crop_sizes[self.progressive_idx]

        self.scheduler.T_max = self.args.end_epoch
        self.criterion_l1 = L1Loss()
        self.start_scheduler_epoch = args.start_scheduler_epoch

        self.set_train_dataloader()
        self.set_val_dataloader()

    def set_train_dataloader(self):
        # Traning loader
        train_data_path = self.args.data_path
        if self.args.only_use_generate_data:
            train_data_path = None
        Train_set = RealBlur_Loader(data_path=train_data_path, mode="train", crop_size=self.now_crop_size)
        train_sampler = DistributedSampler(Train_set)
        dataloader_train = DataLoader(Train_set, sampler=train_sampler, batch_size=self.now_batch_size_per_gpu, num_workers=8, pin_memory=True)
        
        self.dataloader_train = dataloader_train
        self.train_sampler = train_sampler
        
        logging.info(f"####################################")
        logging.info(f"Train Data length:{len(self.dataloader_train.dataset)}")
        logging.info(f"Train Data batch_size:{self.now_batch_size_per_gpu}")
        logging.info(f"Train Data crop size:{self.now_crop_size}")
        logging.info(f"####################################")

    def set_val_dataloader(self):
        # Val loader
        Val_set = RealBlur_Loader(data_path=self.args.data_path, mode="test", crop_size=self.mini_crop_sizes[-1])
        dataloader_val = DataLoader(Val_set, batch_size=self.mini_batch_sizes[-1], shuffle=True, num_workers=8,
                                    drop_last=False, pin_memory=True)
        
        self.dataloader_val = dataloader_val

    def train(self):
        if dist.get_rank() == 0:
            print('###########################################')
            print('Start_Epoch:', self.args.start_epoch)
            print('End_Epoch:', self.args.end_epoch)
            print('Model:', self.args.model_name)
            print(f"Optimizer:{self.optimizer.__class__.__name__}")
            print(f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}")
            print(f"Train Data length:{len(self.dataloader_train.dataset)}")
            print("start train !!")
            print('###########################################')

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            self.epoch = epoch

            #-----------------progressive learning----------------
            if self.epoch > self.threshold_epochs and self.progressive_idx + 1 < len(self.mini_epochs):
                self.progressive_idx += 1
                self.threshold_epochs += self.mini_epochs[self.progressive_idx]
                self.now_batch_size_per_gpu = self.mini_batch_sizes[self.progressive_idx]
                self.now_crop_size = self.mini_crop_sizes[self.progressive_idx]

                self.set_train_dataloader()
            #-----------------------------------------------------
            self._train_epoch()

            if dist.get_rank() == 0:
                if (epoch % self.args.validation_epoch) == 0 or epoch == self.args.end_epoch:
                    self.valid()

                if(self.args.val_save_epochs > 0 and epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch):
                    self.val_save_image(dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset)

                self.save_model()
    
    def _train_epoch(self):
        self.train_sampler.set_epoch(self.epoch)
        tq = tqdm.tqdm(self.dataloader_train, total=len(self.dataloader_train))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] training')
        total_train_loss = AverageMeter()
        total_train_psnr = AverageMeter()
        total_train_lpips = AverageMeter()
        
        for idx, sample in enumerate(tq):
            self.model.train()
            self.optimizer.zero_grad()
             # input: [B, C, H, W], gt: [B, C, H, W]
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            output = self.model(blur)
            output =  output.clamp(-0.5, 0.5)   # [B, C, H, W]
            # Compute loss at each stage
            loss = self.criterion_l1(output, sharp)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_train_loss.update(loss.detach().item())
            psnr = calc_psnr(output.detach(), sharp.detach())
            total_train_psnr.update(psnr)

            tq.set_postfix({'loss': total_train_loss.avg, 'psnr': total_train_psnr.avg, 'lpips': total_train_lpips.avg,'lr': optimizer.param_groups[0]['lr']})

        if self.scheduler and self.epoch > self.start_scheduler_epoch:
            self.scheduler.step()
        
        if self.writer and dist.get_rank() == 0:
            self.writer.add_scalar('Loss/Train_loss', total_train_loss.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_psnr', total_train_psnr.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_lpips', total_train_lpips.avg, self.epoch)
            logging.info(
                f'Epoch [{self.epoch}/{args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_psnr:{total_train_psnr.avg:.4f} Train_lpips:{total_train_lpips.avg:.4f}')
    
    @torch.no_grad()
    def _valid(self, blur, sharp):
        self.model.eval()
        output = self.model(blur)
        output =  output.clamp(-0.5, 0.5)   # [B, C, H, W]
        # Compute loss at each stage
        loss = self.criterion_l1(output, sharp)

        psnr = torch.mean(self.psnr_func(output.detach(), sharp.detach())).item()
        lpips = torch.mean(self.lpips_func(output.detach(), sharp.detach())).item()
        return psnr, lpips, loss.item()
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        total_val_psnr = AverageMeter()
        total_val_lpips = AverageMeter()
        total_val_loss = AverageMeter()
        tq = tqdm.tqdm(self.dataloader_val, total=len(self.dataloader_val))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] Validation')
        for idx, sample in enumerate(tq):
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            psnr, lpips, loss = self._valid(blur, sharp)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            total_val_loss.update(loss)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg, Loss=total_val_loss.avg)

        self.writer.add_scalar('Val/Test_lpips', total_val_lpips.avg, self.epoch)
        self.writer.add_scalar('Val/Test_psnr', total_val_psnr.avg, self.epoch)
        self.writer.add_scalar('Val/Test_loss', total_val_loss.avg, self.epoch)
        logging.info(
            f'Crop Validation Epoch [{self.epoch}/{args.end_epoch}]: Test Loss: {total_val_loss.avg:.4f} Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}')
        
        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            args.best_psnr = self.best_psnr
            best_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_{}.pth'.format(args.model_name)))

            print('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            logging.info('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            
    def save_model(self):
        """save model parameters"""
        training_state = {'epoch': self.epoch, 
                          'model_state': self.model.module.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                          'best_panr': self.best_psnr,
                          'args': args}
        torch.save(training_state, os.path.join(self.args.dir_path, 'last_{}.pth'.format(self.args.model_name)))

        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(training_state, os.path.join(self.args.dir_path, 'epoch_{}_{}.pth'.format(self.epoch, self.args.model_name)))

        if self.epoch == self.args.end_epoch:
            model_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_{}.pth'.format(args.model_name)))

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        """use train set to val and save image"""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur, sharp = sample['blur'].unsqueeze(0).to(device), sample['sharp'].unsqueeze(0).to(device)
            output = self.model(blur) # [3, C, H, W]
            output = output.clamp(-0.5, 0.5) # [C, H, W]

            save_img_dir_path = os.path.join(dir_path, f'visualization', 'output')
            os.makedirs(save_img_dir_path, exist_ok=True)
            save_sharp_dir_path = os.path.join(dir_path, f'visualization', 'sharp')
            os.makedirs(save_sharp_dir_path, exist_ok=True)

            save_img_path = os.path.join(save_img_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            output = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output)

            save_sharp_path = os.path.join(save_sharp_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            sharp = tensor2cv(sharp + 0.5)
            cv2.imwrite(save_sharp_path, sharp)

if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=500, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--mini_epochs", default=[500], type=int, nargs='+')
    parser.add_argument("--mini_batch_sizes", default=[1], type=int, nargs='+')
    parser.add_argument("--mini_crop_sizes", default=[384], type=int, nargs='+')
    parser.add_argument("--start_scheduler_epoch", default=0, type=int)
    parser.add_argument("--validation_epoch", default=50, type=int)
    parser.add_argument("--check_point_epoch", default=100, type=int)
    parser.add_argument("--init_lr", default=3e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default='adamw', type=str)
    parser.add_argument("--criterion", default='l1', type=str)
    parser.add_argument("--data_path", default='./dataset/Realblur_J', type= str)
    parser.add_argument("--generate_path", default=None, type=str, nargs='+')
    parser.add_argument("--dir_path", default='./experiments/Restormer_Realblur_J', type=str)
    parser.add_argument("--model_name", default='Restormer_Realblur_J', type=str)
    parser.add_argument("--model", default='Restormer', type=str, choices=['Restormer'])
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--val_save_epochs", default=100, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--only_use_generate_data", action='store_true', help="only use generated data to train model.")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')

    net = Restormer()

    # training seed
    seed = args.seed + args.local_rank
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # device
    args.device = device
    print("device:", device)
    num_gpus = torch.cuda.device_count()
    net.to(device)

    print(args.__dict__.items())

    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.init_lr, weight_decay=1e-4, betas=[0.9, 0.999])
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch - args.start_scheduler_epoch, eta_min=args.min_lr)
    
    # load pretrained
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    if os.path.exists(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))):
        print('load_pretrained')
        training_state = (torch.load(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name)), map_location=map_location))
        args.start_epoch = training_state['epoch'] + 1
        if 'best_psnr' in training_state['args']:
            args.best_psnr = training_state['args'].best_psnr
        new_weight = net.state_dict()
        training_state["model_state"] = judge_and_remove_module_dict(training_state["model_state"])
        new_weight.update(training_state['model_state'])
        net.load_state_dict(new_weight)
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state['optimizer_state'])
        optimizer.load_state_dict(new_optimizer)
        new_scheduler = scheduler.state_dict()
        new_scheduler.update(training_state['scheduler_state'])
        scheduler.load_state_dict(new_scheduler)
    elif args.resume:
        print('load_resume_pretrained')
        model_load = torch.load(args.resume, map_location=map_location)
        if 'model_state' in model_load.keys():
            model_load["model_state"] = judge_and_remove_module_dict(model_load["model_state"])
            net.load_state_dict(model_load['model_state'])
        else:
            model_load = judge_and_remove_module_dict(model_load)
            net.load_state_dict(model_load)
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    # Model
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)

    
    writer = None
    if dist.get_rank() == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, 'train.log') , format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
        
        logging.info(f'args: {args}')
        logging.info(f'model: {net}')
        logging.info(f'model parameters: {count_parameters(net)}')
        logging.info(f"Optimizer:{optimizer.__class__.__name__}")

        writer = SummaryWriter(os.path.join("Restormer_log", args.model_name))
        writer.add_text("args", str(args))

    trainer = Trainer(net, optimizer, scheduler, args, writer)
    trainer.train()
    

    

