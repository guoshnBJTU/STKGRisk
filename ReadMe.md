# Running
- Command for Manhatton Dataset
``` python
python -u train.py --inverse --depth_gcn --pretrain_kg --config config/manhatton/STKGRisk_MANHATTON_Config_v5_multi.json --aug  --part_top_mask --gpus 0,1 > output/man.out
```
- Command for Brooklyn Dataset
``` Python
python -u train.py --inverse --depth_gcn --pretrain_kg --config config/brooklyn/STKGRisk_BROOKLYN_Config_v5_multi.json --aug --part_top_mask --gpus 0,1,2 > output/bro.out
```