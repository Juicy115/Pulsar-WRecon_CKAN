nohup python3 main.py --model WGAN-GP \
               --is_train True \
               --dataset htru \
               --generator_iters 9000 \
               --cuda True \
               --batch_size 64 &> htru.log &