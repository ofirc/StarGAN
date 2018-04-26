#!/bin/bash
python main.py \
  --mode=train \
  --dataset=CelebA \
  --c_dim=5 \
  --image_size=128 \
  --sample_path=stargan_celebA/samples_11_apr \
  --log_path=stargan_celebA/logs_11_apr \
  --model_save_path=stargan_celebA/models_11_apr \
  --result_path=stargan_celebA/results_11_apr 
#  --pretrained_model=15_12000
