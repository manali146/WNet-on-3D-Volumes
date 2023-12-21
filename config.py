# Hyperparameters
import torch

class Config():
	def __init__(self):
		self.LEARNING_RATE = 1e-4
		self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
		#self.input_size = 96 # Side length of square image patch
		self.batch_size = 1 # Batch size of patches Note: 11 gig gpu will max batch of 5
		self.in_chans = 1
		self.out_chans = 5
		self.ch_mul=64
		self.num_epochs = 350 #250 for real
		self.NUM_WORKERS = 2
		self.k = 4 # Number of classes
		self.TRAIN_IMG_DIR = "/mnt/Data2/rad_data_science/SpectralFingerprinting/DL/data/train_images/" # Directory of images
		#self.VAL_IMG_DIR = "DL/data/val_images/"

		self.showdata = False # Debug the data augmentation by showing the data we're training on.
		self.useInstanceNorm = True # Instance Normalization
		self.useBatchNorm = False # Only use one of either instance or batch norm
		self.useDropout = True
		self.drop = 0.2

# Each item in the following list specifies a module.
# Each item is the number of input channels to the module.
# The number of output channels is 2x in the encoder, x/2 in the decoder.
		self.encoderLayerSizes = [64, 128, 256]
		self.decoderLayerSizes = [512, 256]
		#self.showSegmentationProgress = True
		#self.segmentationProgressDir = './latent_images/'
		#self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine
		self.saveModel = True
    	
    	
    	
    	
    	
    	
