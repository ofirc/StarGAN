#!/bin/bash
python main.py --mode='test' --dataset='CelebA' --c_dim=5 --image_size=128 --test_model='17_7000' \
               --sample_path='stargan_celebA/samples' --log_path='stargan_celebA/logs' \
               --model_save_path='stargan_celebA/models_9_apr' --result_path='stargan_celebA/results'
