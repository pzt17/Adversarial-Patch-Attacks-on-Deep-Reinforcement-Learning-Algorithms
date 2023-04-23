import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2)
		)
		
		self.conv_mid = nn.Conv2d(256, 1000, 1)
		
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1000, 256, kernel_size = 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1),
			nn.BatchNorm2d(3),
			nn.Sigmoid()
		)
		
		
	
	def forward(self, x):
		
		x = 1.0*x*(1.0/255.0)
		x_enc = self.encoder(x)
		x_mid = self.conv_mid(x_enc)
		x_generate = self.decoder(x_mid)
		x_generate = 1.0*x_generate*255.0
		
		return x_generate
	
