CUDA_VISIBLE_DEVICES=4 python main.py --split val_unseen --eval 1 --load pretrained_models/sem_exp.pth \
               --num_processes_per_gpu 1 --num_processes_on_first_gpu -3 \
               -v 2 -n 1 --print_images 1 -d results/ --exp_name exp2 --camera_height 0.88