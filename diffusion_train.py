import argparse
import torch
import torch.nn as nn
import os
import logging
import math
from torchvision.utils import save_image
from itertools import islice
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import cv2
import sys
from dataloader import Flow_Loader
from models.diffusion_model import UNet
from models.diffusion_network import DDIM
from utils.utils import same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
import pyiqa
from torch.utils.tensorboard import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, dataloader_train, dataloader_val, model, optimizer, scheduler, args, writer) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.sample_timesteps = self.args.sample_timesteps
        self.device = self.args.device
        self.psnr_func = pyiqa.create_metric('psnr', device=device)
        self.lpips_func = pyiqa.create_metric('lpips', device=device)
        self.best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0
        self.grad_clip = 1
        
    def train(self):
        print('Start_Epoch:', self.args.start_epoch)
        print('End_Epoch:', self.args.end_epoch)
        print('Model:', self.args.model_name)
        print(f"Optimizer:{self.optimizer.__class__.__name__}")
        print(f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}")
        print("start train")

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            if (epoch % self.args.validation_epoch) == 0 or epoch == self.args.end_epoch:
                self.valid()

            if(self.args.val_save_epochs > 0 and epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch):
                self.val_save_image(dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset)

            self.save_model()
    
    def _train_epoch(self):
        tq = tqdm.tqdm(dataloader_train, total=len(dataloader_train))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] training')
        total_train_loss = AverageMeter()
        total_train_psnr = AverageMeter()
        total_train_lpips = AverageMeter()
        
        for idx, sample in enumerate(tq):
            self.model.train()
            self.optimizer.zero_grad()
 
            blur, sharp = sample['blur'].to(self.device), sample['sharp'].to(self.device)
            flow = sample['flow'].to(self.device)
            condition = torch.cat([sharp, flow], dim=1)
            loss = self.model(x=blur, condition=condition)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_train_loss.update(loss.detach().item())

            # if idx % 10 == 0:
            #     psnr, lpips = self._valid(blur, gt)
            #     total_train_psnr.update(psnr)
            #     total_train_lpips.update(lpips)

            tq.set_postfix({'loss': total_train_loss.avg, 'psnr': total_train_psnr.avg, 'lpips': total_train_lpips.avg,'lr': optimizer.param_groups[0]['lr']})

        if self.scheduler:
            self.scheduler.step()
        self.writer.add_scalar('Loss/Train_loss', total_train_loss.avg, self.epoch)
        self.writer.add_scalar('Loss/Train_psnr', total_train_psnr.avg, self.epoch)
        self.writer.add_scalar('Loss/Train_lpips', total_train_lpips.avg, self.epoch)
        logging.info(
            f'Epoch [{self.epoch}/{args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_psnr:{total_train_psnr.avg:.4f} Train_lpips:{total_train_lpips.avg:.4f}')
    
    @torch.no_grad()
    def _valid(self, sharp, blur, flow):
        self.model.eval()
        condition = torch.cat([sharp, flow], dim=1)
        output = self.model.sample(condition=condition, sample_timesteps=self.sample_timesteps, device=self.device)
        output = output.clamp(-0.5, 0.5)
        psnr = torch.mean(self.psnr_func(output.detach(), blur.detach())).item()
        lpips = torch.mean(self.lpips_func(output.detach(), blur.detach())).item()
        return psnr, lpips
    
    @torch.no_grad()
    def valid(self, valid_iters=10):
        self.model.eval()
        total_val_psnr = AverageMeter()
        total_val_lpips = AverageMeter()
        tq = tqdm.tqdm(islice(self.dataloader_val, valid_iters), total=valid_iters)
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] Validation')
        for idx, sample in enumerate(tq):
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            flow = sample['flow'].to(self.device)
            psnr, lpips = self._valid(sharp, blur, flow)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg)

        self.writer.add_scalar('Val/Test_lpips', total_val_lpips.avg, self.epoch)
        self.writer.add_scalar('Val/Test_psnr', total_val_psnr.avg, self.epoch)
        logging.info(
            f'Validation Epoch [{self.epoch}/{args.end_epoch}]: Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}')
        
        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            args.best_psnr = self.best_psnr
            best_state = {'model_state': self.model.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_{}.pth'.format(args.model_name)))

            print('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            logging.info('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            
    def save_model(self):
        """save model parameters"""
        training_state = {'epoch': self.epoch, 
                          'model_state': self.model.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                          'best_panr': self.best_psnr,
                          'args': args}
        torch.save(training_state, os.path.join(self.args.dir_path, 'last_{}.pth'.format(self.args.model_name)))

        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(training_state, os.path.join(self.args.dir_path, 'epoch_{}_{}.pth'.format(self.epoch, self.args.model_name)))

        if self.epoch == self.args.end_epoch:
            model_state = {'model_state': self.model.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_{}.pth'.format(args.model_name)))

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        """use train set to val and save image"""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur, sharp = sample['blur'].unsqueeze(0).to(device), sample['sharp'].unsqueeze(0).to(device)
            flow = sample['flow'].unsqueeze(0).to(self.device)
            condition = torch.cat([sharp, flow], dim=1)
            output = self.model.sample(condition=condition, sample_timesteps=self.sample_timesteps, device=self.device)
            output = output.clamp(-0.5, 0.5)

            save_img_dir_path = os.path.join(dir_path, f'visualization', 'output')
            os.makedirs(save_img_dir_path, exist_ok=True)
            save_sharp_dir_path = os.path.join(dir_path, f'visualization', 'sharp')
            os.makedirs(save_sharp_dir_path, exist_ok=True)
            save_blur_dir_path = os.path.join(dir_path, f'visualization', 'blur')
            os.makedirs(save_blur_dir_path, exist_ok=True)

            save_img_path = os.path.join(save_img_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            output = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output)

            save_sharp_path = os.path.join(save_sharp_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            sharp = tensor2cv(sharp + 0.5)
            cv2.imwrite(save_sharp_path, sharp)

            save_blur_path = os.path.join(save_blur_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            blur = tensor2cv(blur + 0.5)
            cv2.imwrite(save_blur_path, blur)

def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch",default=5000,type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--crop_size", default=128, type=int)
    parser.add_argument("--init_lr",default=1e-4,type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--beta_1",default=1e-6,type=float)
    parser.add_argument("--beta_T",default=1e-2,type=float)
    parser.add_argument("--dropout",default=0.0,type=float)
    parser.add_argument("--weight_decay",default=0,type=float)
    parser.add_argument("--num_timesteps",default=2000,type=int)
    parser.add_argument("--dir_path",default="./experiments/ID_Blau",type=str)
    parser.add_argument("--data_path",default="./dataset/GOPRO_Large",type=str)
    parser.add_argument("--flow_data_path",default="./dataset/GOPRO_flow",type=str)
    parser.add_argument("--flow_norm",default=True,type=bool)
    parser.add_argument("--model_name", default='ID_Blau', type=str)
    parser.add_argument("--model", default='UNet', choices=['UNet'], type=str)
    parser.add_argument("--optimizer",default="adam",type=str)
    parser.add_argument("--opt_beta1",default=0.9,type=float)
    parser.add_argument("--scheduler",default=None, type=str)
    parser.add_argument("--sample_timesteps",default=20, type=int)
    parser.add_argument("--base_channels",default=64, type=int)
    parser.add_argument("--time_dim",default=256, type=int)
    parser.add_argument("--channel_mults",default=(1, 2, 3), type=int, nargs='+')
    parser.add_argument("--num_res_blocks",default=2, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--validation_epoch", default=50, type=int)
    parser.add_argument("--val_save_epochs", default=50, type=int)
    parser.add_argument("--check_point_epoch", default=200, type=int)
    parser.add_argument("--criterion", default='l1', type=str)
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device :",device)
    #same_seed(args.seed)
    print(args.__dict__.items())

    # Traning loader
    Train_set = Flow_Loader(data_path=args.data_path,
                            flow_path=args.flow_data_path,
                            mode="train",
                            crop_size=args.crop_size,
                            flow_norm=args.flow_norm)
    dataloader_train = DataLoader(Train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                drop_last=False)
    # Valing loader
    Val_set = Flow_Loader(data_path=args.data_path,
                            flow_path=args.flow_data_path,
                            mode="test",
                            crop_size=None,
                            flow_norm=args.flow_norm)
    dataloader_val = DataLoader(Val_set, batch_size=1, shuffle=True, num_workers=8,
                                drop_last=False)
    
    beta = generate_linear_schedule(args.num_timesteps, args.beta_1, args.beta_T)

    if args.model == 'UNet':
        net = UNet(
            img_channels=9,
            base_channels=args.base_channels,
            channel_mults=args.channel_mults, 
            time_dim=args.time_dim,
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout
            ).to(device)
    else:
        raise ValueError("model error")
    
    diffusionModel = DDIM(net, img_channels=9, betas=beta, criterion=args.criterion).to(device)

    # diffusionModel = nn.DataParallel(diffusionModel)
    # diffusionModel.to(device)
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.init_lr, betas=(args.opt_beta1, 0.999))
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch, eta_min=args.min_lr)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"scheduler not supported {args.scheduler}")
    # load pretrained
    if os.path.exists(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))):
        print('load_last_pretrained')
        training_state = (torch.load(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))))
        args.start_epoch = training_state['epoch'] + 1
        if 'best_psnr' in training_state['args']:
            args.best_psnr = training_state['args'].best_psnr
        training_state['model_state'] = judge_and_remove_module_dict(training_state['model_state'])
        new_weight = diffusionModel.state_dict()
        new_weight.update(training_state['model_state'])
        diffusionModel.load_state_dict(new_weight)
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state['optimizer_state'])
        optimizer.load_state_dict(new_optimizer)
        if scheduler:
            new_scheduler = scheduler.state_dict()
            new_scheduler.update(training_state['scheduler_state'])
            scheduler.load_state_dict(new_scheduler)
    elif args.resume:
        print('load_resume_pretrained')
        model_load = torch.load(args.resume)
        if 'model_state' in model_load.keys():
            diffusionModel.load_state_dict(model_load['model_state'])
        else:
            diffusionModel.load_state_dict(model_load)
        os.makedirs(args.dir_path)
    else:
        os.makedirs(args.dir_path)

    logging.basicConfig(
        filename=os.path.join(args.dir_path, 'train.log') , format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
    
    logging.info(f'args: {args}')
    logging.info(f'model: {diffusionModel}')
    logging.info(f'model parameters: {count_parameters(diffusionModel)}')
    logging.info(f"Optimizer:{optimizer.__class__.__name__}")
    logging.info(f"Scheduler:{scheduler.__class__.__name__ if scheduler else None}")

    writer = SummaryWriter(os.path.join("log", args.model_name))
    writer.add_text("args", str(args))

    trainer = Trainer(dataloader_train, dataloader_val, diffusionModel, optimizer, scheduler, args, writer)
    trainer.train()