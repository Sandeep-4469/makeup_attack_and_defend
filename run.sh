

# For makeup removal
cd makeup_removal
#Data loading and Training of GAN
python train.py
#for inference change file paths in the inference.py
python inference.py 

# For BeautyGAN attack (transferability)
cd makeup_attack
python train_beautygan.py --dataroot_A ./data/non-makeup     --dataroot_B ./data/makeup     --name beautygan_makeup_run     --model cycle_gan

#For testing with single image
python test_image.py # change image paths in the python file