#!/bin/bash
python main.py \
  --mode=train \
  --dataset=CelebA \
  --c_dim=5 \
  --image_size=128 \
  --sample_path=stargan_celebA/samples_29_apr_dist_loss \
  --log_path=stargan_celebA/logs_29_apr_dist_loss \
  --model_save_path=stargan_celebA/models_29_apr_dist_loss \
  --result_path=stargan_celebA/results_29_apr_dist_loss 
#  --pretrained_model=15_12000
