# Running
- Command for Brooklyn Dataset
``` Python
python -u train.py --inverse --depth_gcn --pretrain_kg --config config/brooklyn/STKGRisk_BROOKLYN_Config_v5_multi.json --aug --part_top_mask --gpus 0,1 > output/bro.out
```