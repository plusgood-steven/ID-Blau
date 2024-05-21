# Preparing Blur Condition

We use
[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf) to esimate flow optical.

We have provided pretrained models (raft-things.pth) under the "./weights" path.

## Generating Condition

Run the code to generate blur condtions.

```Shell
cd PrepareCondition
python generate_condition.py --mode all --model=weights/raft-things.pth --dir_path=../dataset/GOPRO_flow
```
