# ddpm-from-scratch


python train.py --workers 0 --batch_size 64 --lr 1e-4 --no_amp

# To generete images

python sample.py --ckpt checkpoints/step_0036000.pt --use_ema

# To contmue traning: 
python train.py --workers 0 --batch_size 64 --lr 1e-4 --no_amp


python train.py --epochs 100 --workers 0 --batch_size 64 --lr 1e-4 --no_amp
