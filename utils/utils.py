import numpy as np
import math
import torch
import random
import cv2

def calc_psnr(result, gt):
    result = (result + 0.5) * 255
    gt = (gt + 0.5) * 255
    result = result[0].cpu().numpy()
    gt = gt[0].cpu().numpy()
    mse = np.mean(np.power((result / 255. - gt / 255.), 2))
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor2cv(input: torch.Tensor):
    input = input.clone().detach().cpu().squeeze()
    input = input.mul_(255).add_(0.5).clamp_(0,255).permute(1, 2, 0).type(torch.uint8).numpy()
    input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)

    return input

def create_gaussian_noise(image_tensor_size, mean=0, std=1, num_channels=4):
    """
    Create Gaussian noise.

    Args:
        image_tensor_size: a size of shape (B, C, H, W), where B is the batch size,
                      C is the number of channels, H is the height, and W is the width.
        mean (optional): the mean of the Gaussian distribution. Default is 0.
        std (optional): the standard deviation of the Gaussian distribution. Default is 1.
        num_channels (optional): the number of channels of the noise tensor. Default is 4.

    Returns:
        a new PyTorch tensor of shape (B, num_channels, H, W) representing the noise, where B is the original batch size.
    """
    b, c, h, w = image_tensor_size

    # create Gaussian noise with given mean and standard deviation
    noise_shape = [b, num_channels, h, w]
    noise = torch.randn(*noise_shape) * std + mean

    return noise

def concat_noise(image_tensor, noise, num_dim=1):
    """
    Concat Gaussian noise to a given PyTorch tensor representing an image.

    Args:
        image_tensor: a PyTorch tensor of shape (B, C, H, W) representing an image, where B is the batch size,
                      C is the number of channels, H is the height, and W is the width.
        mean (optional): the mean of the Gaussian distribution. Default is 0.
        std (optional): the standard deviation of the Gaussian distribution. Default is 1.
        num_dim (optional): the number of dimensions along which to concat noise. Default is 1.

    Returns:
        a new PyTorch tensor of shape (B, C + num_channels, H, W) representing the noisy image, where B is the original batch size.
    """
    assert image_tensor.size()[0] == noise.size()[0]
    noise = noise.clone().to(image_tensor.device)
    # concatenate the noise tensor with the original image tensor along the specified dimensions
    noisy_image_tensor = torch.cat([image_tensor, noise], dim=num_dim)

    # clip the pixel values to [-0.5, 0.5]
    noisy_image_tensor = torch.clamp(noisy_image_tensor, -0.5, 0.5)

    return noisy_image_tensor

def create_concat_noise(image_tensor, mean=0, std=1, num_channels=4, num_dim=1):
    """
    Create and concat Gaussian noise to a given PyTorch tensor representing an image.

    Args:
        image_tensor: a PyTorch tensor of shape (B, C, H, W) representing an image, where B is the batch size,
                      C is the number of channels, H is the height, and W is the width.
        mean (optional): the mean of the Gaussian distribution. Default is 0.
        std (optional): the standard deviation of the Gaussian distribution. Default is 1.
        num_channels (optional): the number of channels of the noise tensor. Default is 4.
        num_dim (optional): the number of dimensions along which to concat noise. Default is 1.

    Returns:
        a new PyTorch tensor of shape (B, C + num_channels, H, W) representing the noisy image, where B is the original batch size.
    """
    b, c, h, w = image_tensor.size()

    # create Gaussian noise with given mean and standard deviation
    noise_shape = [b, num_channels, h, w]
    noise = torch.randn(*noise_shape) * std + mean

    noise = noise.to(image_tensor.device)
    # concatenate the noise tensor with the original image tensor along the specified dimensions
    noisy_image_tensor = torch.cat([image_tensor, noise], dim=num_dim)

    # clip the pixel values to [-0.5, 0.5]
    noisy_image_tensor = torch.clamp(noisy_image_tensor, -0.5, 0.5)

    return noisy_image_tensor

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def shuffle_tensor_dim(tensor, dim):
    size = tensor.size()
    index = list(range(size[dim]))
    np.random.shuffle(index)
    index = torch.LongTensor(index).to(tensor.device)
    return tensor.index_select(dim, index)

def judge_and_remove_module_dict(load_state_dict, remove_key='module.'):
    new_dict = {}
    for old_key, value in load_state_dict.items():
        if old_key.startswith(remove_key):
            new_key = old_key.replace(remove_key, '')
            new_dict[new_key] = value
        else:
            new_dict[old_key] = value
    return new_dict