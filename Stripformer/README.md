# Training of Stripformer

We train Stripformer with ID-Blau in two stages:

Relative paths should be defined based on the root directory path.

- We pre-train Stripformer for 1000 epochs.
- Run the following command.

  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 Stripformer/deblur_train_pretrained.py --only_use_generate_data --generate_path ./dataset/GOPRO_Large_Reblur
  ```

After pretraining, we proceed with finetuning based on the original configuration of Stripformer.

- We use the pretrained weights from the 500th epoch for finetuning to avoid excessively long training times. Alternatively, you can choose weights from later epochs if needed.
- First use a patch size of 256x256 to train Stripformer for 3000 epochs.
- Run the following command.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 Stripformer/deblur_train.py --resume ./experiments/Stripformer_pretrained/epoch_500_Stripformer_pretrained.pth
```

- After 3000 epochs, we keep training Stripformer for 1000 epochs with patch size 512x512
- Run the following command.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 Stripformer/deblur_train.py --resume ./experiments/Stripformer_first_stage/final_Stripformer_first_stage.pth
```
