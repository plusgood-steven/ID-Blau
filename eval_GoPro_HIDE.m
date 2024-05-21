close all;clear all;

% Reminder: Change the following parameters as needed;
model = {'MIMO_UnetPlus'} % Change model here
datasets = {'GoPro', 'HIDE'}; % Change datasets here
num_set = length(datasets);

tic
delete(gcp('nocreate'))
parpool('local',20);

for idx_set = 1:num_set
    file_path = strcat('./out/', model, '/results/', datasets{idx_set}, '/');
    gt_path = strcat('./dataset/test/', datasets{idx_set}, '/target/');
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        parfor j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    
    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

end
delete(gcp('nocreate'))
toc