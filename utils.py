import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import PhantomDataset
import numpy as np
from scipy.stats import norm
import os
import nibabel as nib
import matplotlib.pyplot as plt

def save_checkpoint(state, filename= "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    #val_dir,
    batch_size,
    train_transform,
    #val_transform,
    num_workers=4
):
    train_ds = PhantomDataset(
        image_dir=train_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    # val_ds = CarvanaDataset(
    #     image_dir=val_dir,
    #     mask_dir=val_maskdir,
    #     transform=val_transform,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    return train_loader#, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

# def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
def save_segmented_images(enc_output, dec_output, model, data_loader, epoch, batch, folder, device):
	model.eval()
	save_dir = os.path.join(folder, f"epoch_{epoch}")
	os.makedirs(save_dir, exist_ok=True)
	#original_affine = enc_output['affine']

	with torch.no_grad():
		segmentation = torch.argmax(enc_output, dim=1).cpu().numpy()
		img = nib.Nifti1Image(segmentation[0], affine=np.eye(4), dtype=np.uint16)
		#nib.save(img, os.path.join(save_dir, f"seg_{batch}_enc.nii.gz"))
		segmentation = torch.argmax(dec_output, dim=1).cpu().numpy()
		img = nib.Nifti1Image(segmentation[0], affine=np.eye(4), dtype=np.uint16)
		nib.save(img, os.path.join(save_dir, f"seg_{batch}_dec.nii.gz"))
		# Print in the terminal
		#img_data = img.get_fdata()
		#slice_index = 20
		#plt.imshow(img_data[:, :, slice_index], cmap='gray')
		#plt.show()

	model.train()

def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    

def gaussian_kernel_3d(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1, 1) + x_2.reshape(1, -1, 1) + x_2.reshape(1, 1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float16))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel
    
    
