# ddpm-from-scratch


python train.py --workers 0 --batch_size 64 --lr 1e-4 --no_amp

# To generete images

python sample.py --ckpt checkpoints/epoch_042_step_0030934.pt --use_ema

# To contmue traning: 
python train.py --workers 0 --batch_size 64 --lr 1e-4 --no_amp
