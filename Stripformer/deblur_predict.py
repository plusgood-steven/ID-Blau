import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import sys
import tqdm
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import Test_Loader
from Stripformer.model import get_nets
from utils.utils import same_seed, count_parameters, judge_and_remove_module_dict


@torch.no_grad()
def predict(model, args, device):
    model.eval()

    if args.dataset == 'GoPro+HIDE':
        dataset_name = ['GoPro', 'HIDE']
    else:
        dataset_name = [args.dataset]

    for val_dataset_name in dataset_name:
        dataset_path = os.path.join(args.data_path, val_dataset_name)

        dataset = Test_Loader(data_path=dataset_path,
                                crop_size=args.crop_size,
                                ZeroToOne=False)
        save_dir = os.path.join(args.dir_path, 'results', f'{val_dataset_name}')
        os.makedirs(save_dir)
        dataset_len = len(dataset)
        tq = tqdm.tqdm(range(dataset_len))
        tq.set_description(f'Predict {val_dataset_name}')

        for idx in tq:
            sample = dataset[idx]
            input = sample['blur'].unsqueeze(0).to(device)
            b, c, h, w = input.shape
            factor=8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            input = torch.nn.functional.pad(input, (0, w_n, 0, h_n), mode='reflect')

            output = model(input)
            output = output[:, :, :h, :w]
            output = output.clamp(-0.5, 0.5)

            image_name = os.path.split(dataset.get_path(idx=idx)['blur_path'])[-1]
            save_img_path = os.path.join(save_dir, image_name)

            save_image(output.squeeze(0).cpu() + 0.5, save_img_path)



if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/test', type=str)
    parser.add_argument("--dir_path", default='./out/Stripformer', type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model", default='Stripformer', type=str, choices=['Stripformer'])
    parser.add_argument("--dataset", default='GoPro+HIDE', type=str, choices=['GoPro+HIDE', 'GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R', 'RWBI'])
    parser.add_argument("--crop_size", default=None, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    load_model_state = torch.load(args.model_path)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    # Model and optimizer
    net = get_nets(args.model)
    
    
    load_model_state = torch.load(args.model_path)

    if 'model_state' in load_model_state.keys():
        load_model_state["model_state"] = judge_and_remove_module_dict(load_model_state["model_state"])
        net.load_state_dict(load_model_state["model_state"])
    elif 'model' in load_model_state.keys():
        load_model_state["model"] = judge_and_remove_module_dict(load_model_state["model"])
        net.load_state_dict(load_model_state["model"])
    else:
        load_model_state = judge_and_remove_module_dict(load_model_state)
        net.load_state_dict(load_model_state)

    net = nn.DataParallel(net)
    net.to(device)

    print("device:", device)
    print(f'args: {args}')
    print(f'model: {net}')
    print(f'model parameters: {count_parameters(net)}')

    same_seed(2023)
    predict(net, args=args, device=device)





