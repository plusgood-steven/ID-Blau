import os
from skimage import io
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import concurrent.futures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--gt_path", type=str)
args = parser.parse_args()

out_path = args.out_path
gt_path = args.gt_path

def image_align(deblurred, gt):
    # this function is based on kohler evaluation code
    z = deblurred
    c = np.ones_like(z)
    x = gt

    zs = (np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    termination_eps = 0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
                                             warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

    target_shape = x.shape
    shift = warp_matrix

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT)

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
    # this function is based on skimage.metrics.peak_signal_noise_ratio
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=-1, gaussian_weights=True,
                                               use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR,SSIM)

total_psnr = []
total_ssim = []
out_images = []
gt_images = []

for idx, image_name in enumerate(os.listdir(out_path)):
    out_image_dir = os.path.join(out_path, image_name)
    gt_image_dir = os.path.join(gt_path, image_name)
    out_images.append(out_image_dir)
    gt_images.append(gt_image_dir)

img_files =[(i, j) for i, j in zip(gt_images, out_images)]
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
        total_psnr.append(PSNR_SSIM[0])
        total_ssim.append(PSNR_SSIM[1])
        print(len(total_psnr), f'Avg. PSNR: {sum(total_psnr)/len(total_psnr)}', f'Avg. SSIM: {sum(total_ssim)/len(total_ssim)}')

print('Avg. PSNR:', sum(total_psnr)/len(total_psnr))
print('Avg. SSIM:', sum(total_ssim)/len(total_ssim))
