#import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import preprocessing_transform
import random
from config import Config
from model_129_111_80 import WNet
from soft_n_cut_loss import soft_n_cut_loss
from loss import NCutLoss3D, OpeningLoss3D
import wandb
wandb.init(project="WNet", name="24_32imgs_e350_O5")
from utils import (
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	check_accuracy,
	save_segmented_images
)

config = Config()
ncut_loss_fn = NCutLoss3D(radius=4, sigma_1=5, sigma_2=1)

def reconstruction_loss(inputs, reconstructed):
	#binary_cross_entropy = F.binary_cross_entropy(inputs, reconstructed, reduction='sum')
	reconstruction_loss = F.mse_loss(inputs, reconstructed, reduction='mean')
	return reconstruction_loss
    	
def train_fn(loader, model, optimizer, loss_fn, epoch):
	#loop = tqdm(loader)
	running_loss = 0.0
	for batch_idx, data in enumerate(loader):
    	
		input_data = preprocessing_transform(data, config.DEVICE)
		input_data = input_data.to(device=config.DEVICE, dtype=torch.float16)
		downsample_factor = 0.5 
		print('INPUT shape--------------->',input_data.shape)
		downsampled_inputs = F.interpolate(input_data, 
		scale_factor=downsample_factor, mode='trilinear', 
		align_corners=False, 
		recompute_scale_factor=True)
		#print('DOWNSAMPLED inputs shape--------------->',downsampled_inputs.shape)
		# forward
		with torch.cuda.amp.autocast():
			enc_output, dec_output = model(input_data) #model(downsampled_inputs)
			# Compute the losses
			l_soft_n_cut = ncut_loss_fn(input_data, enc_output) # with soft_n_cut_loss # (downsampled_inputs, enc_output)
			print('dec_output',dec_output.shape)
			l_reconstruction = reconstruction_loss(input_data, dec_output) #(downsampled_inputs, dec_output) 
			loss = l_reconstruction + l_soft_n_cut # with soft_n_cut_loss

		# backward pass and optimization steps
		optimizer.zero_grad()
		loss.backward(retain_graph=False)
		optimizer.step()
        
		#scaler.scale(loss).backward()
		#scaler.step(optimizer)
		#scaler.update()

		# update loop
		running_loss += loss.item()
		
		# Log metrics to wandb after each batch
		wandb.log({"batch_loss": loss.item(), 
		"soft_n_cut_loss": l_soft_n_cut.item(), # with soft_n_cut_loss
		"reconstruction_loss": l_reconstruction.item()})
		
		# Save segmented 3D images at a specified interval (e.g., every 10 epochs)
		if ((epoch + 1) % 10 == 0) | ((epoch + 1) == 1):
			save_segmented_images(enc_output, dec_output, model, loader, epoch, batch_idx,
			# without_sncl # with_sncl
			folder="/mnt/Data2/rad_data_science/SpectralFingerprinting/DL/saved_images/with_sncl_O5/", # with soft_n_cut_loss
			device=config.DEVICE)
	
	# Log metrics to wandb after each epoch
	wandb.log({"epoch_loss": running_loss / len(loader)})

def main():
	train_transform = A.Compose(
	[
		#A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
		#A.Rotate(limit=35, p=1.0),
		#A.HorizontalFlip(p=0.5),
		#A.VerticalFlip(p=0.1),
		ToTensorV2(),
	],
	)

	model = WNet(k=config.k, ch_mul=config.ch_mul, in_chans=config.in_chans, out_chans=config.out_chans).to(config.DEVICE)
	print('IN channel--------------->',config.in_chans)
	print('OUT channel--------------->',config.out_chans)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

	#train_loader, val_loader = get_loaders(
	train_loader = get_loaders(
		config.TRAIN_IMG_DIR,
		#VAL_IMG_DIR,
		config.batch_size,
		train_transform,
		config.NUM_WORKERS
	)

	scaler = torch.cuda.amp.GradScaler()
    
	for epoch in range(config.num_epochs):
		train_fn(train_loader, model, optimizer, loss_fn, epoch)
		# save model
		checkpoint={
		"state_dict": model.state_dict(),
		"optimizer":optimizer.state_dict(),
		}
        
		#save_checkpoint(checkpoint)
        
		# check accuracy
		# print some examples to a folder
		

	wandb.finish()
    
    
if __name__== "__main__":
	main()


