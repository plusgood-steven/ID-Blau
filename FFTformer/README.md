# Training of FFTformer
We train FFTformer with ID-Blau in two stages:

Relative paths should be defined based on the root directory path.
- We pre-train FFTformer for 1000 epochs.
- Run the following command.

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 FFTformer/deblur_train_pretrained.py --only_use_generate_data --generate_path ./dataset/GOPRO_Large_Reblur
    ```

After pretraining, we proceed with finetuning based on the original configuration of FFTformer. 
- We use the pretrained weights from the 500th epoch for finetuning to avoid excessively long training times. Alternatively, you can choose weights from later epochs if needed.
- Run the following command.

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 FFTformer/deblur_train.py --resume ./experiments/FFTformer_pretrained/epoch_500_FFTformer_pretrained.pth
    ```