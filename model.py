#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:00:00 2023
@author: Manali Thakur

Implementation of the W-Net unsupervised image segmentation architecture
"""

# Import libraries
import torch
import torch.nn as nn
import time
import torch.functional as F


start_time = time.time()

# Set device
device = torch.device("cuda")
torch.cuda.empty_cache()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()        
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm3d(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels)
        self.pointwise_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm3d(out_channels)
        self.depthwise_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels)
        self.pointwise_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

torch.cuda.empty_cache()

class UEnc(nn.Module):
    def __init__(self, k, ch_mul=32, in_chans=1):
        super(UEnc, self).__init__()
        
        self.enc1 = ConvBlock(in_chans, ch_mul) # , seperable=False
        self.enc2 = DepthwiseSeparableConv3D(ch_mul, 2*ch_mul)
        self.enc3 = DepthwiseSeparableConv3D(2*ch_mul, 4*ch_mul, kernel_size=1)
        self.enc4 = DepthwiseSeparableConv3D(4*ch_mul, 8*ch_mul, kernel_size=1)
        
        self.middle = DepthwiseSeparableConv3D(8*ch_mul, 16*ch_mul, kernel_size=1)

        self.dec1 = DepthwiseSeparableConv3D(16*ch_mul, 8*ch_mul, kernel_size=1)
        self.dec2 = DepthwiseSeparableConv3D(8*ch_mul, 4*ch_mul, kernel_size=1)
        self.dec3 = DepthwiseSeparableConv3D(4*ch_mul, 2*ch_mul, kernel_size=1)
        self.dec4 = ConvBlock(2*ch_mul, ch_mul) # , seperable=False
        
        self.upconv1 = nn.ConvTranspose3d(16*ch_mul, 8*ch_mul, kernel_size=(4,4,5), stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose3d(8*ch_mul, 4*ch_mul, kernel_size=4, stride=2, padding=1, output_padding=(0,1,0))
        self.upconv3 = nn.ConvTranspose3d(4*ch_mul, 2*ch_mul, kernel_size=(6,7,6), stride=2, padding=2)
        self.upconv4 = nn.ConvTranspose3d(2*ch_mul, ch_mul, kernel_size=20, stride=2, padding=1, output_padding=(0,1,0))
        
        #self.final = nn.Conv3d(ch_mul, k, kernel_size=1)
        self.final = nn.Sequential(nn.Conv3d(ch_mul, k, 3, 1, 1),nn.Sigmoid(),nn.Softmax(dim=2))

        self.max_pool = nn.MaxPool3d(2)

        
    def forward(self, x):
        conv1_out = self.enc1(x)
        conv2_out = self.enc2(self.max_pool(conv1_out))
        conv3_out = self.enc3(self.max_pool(conv2_out))
        conv4_out = self.enc4(self.max_pool(conv3_out))

        conv5_out = self.middle(self.max_pool(conv4_out))
        #print(self.upconv1(conv5_out).size())
        #print(conv4_out.size())

        conv5_in = torch.cat((self.upconv1(conv5_out), conv4_out), 1)
        conv6_out = self.dec1(conv5_in)
        
        conv6_in = torch.cat((self.upconv2(conv6_out), conv3_out), 1)
        conv7_out = self.dec2(conv6_in)
        
        conv7_in = torch.cat((self.upconv3(conv7_out), conv2_out), 1)
        conv8_out = self.dec3(conv7_in)
        
        conv8_in = torch.cat((self.upconv4(conv8_out), conv1_out), 1)
        conv9_out = self.dec4(conv8_in)
        
        final_out = self.final(conv9_out)
        # padded_seg = self.pad(final_out)
        return final_out

class UDec(nn.Module):
    def __init__(self, squeeze, ch_mul, in_chans, out_chans):
        super(UDec, self).__init__()
        
        self.enc1=ConvBlock(squeeze, ch_mul) # , seperable=False
        self.enc2=DepthwiseSeparableConv3D(ch_mul, 2*ch_mul)
        self.enc3=DepthwiseSeparableConv3D(2*ch_mul, 4*ch_mul, kernel_size=1)
        self.enc4=DepthwiseSeparableConv3D(4*ch_mul, 8*ch_mul, kernel_size=1)
        
        self.middle=DepthwiseSeparableConv3D(8*ch_mul, 16*ch_mul, kernel_size=1)

        self.up1=nn.ConvTranspose3d(16*ch_mul, 8*ch_mul, kernel_size=(4,4,5), stride=2, padding=1)
        
        self.dec1=DepthwiseSeparableConv3D(16*ch_mul, 8*ch_mul, kernel_size=1)
        self.up2=nn.ConvTranspose3d(8*ch_mul, 4*ch_mul, kernel_size=(2,3,2), stride=2)
        
        self.dec2=DepthwiseSeparableConv3D(8*ch_mul, 4*ch_mul, kernel_size=1)
        self.up3=nn.ConvTranspose3d(4*ch_mul, 2*ch_mul, kernel_size=(2,3,2), stride=2)
        
        self.dec3=DepthwiseSeparableConv3D(4*ch_mul, 2*ch_mul, kernel_size=1)
        self.up4=nn.ConvTranspose3d(2*ch_mul, ch_mul, kernel_size=18, stride=2, output_padding=(0,1,0))

        self.dec4=ConvBlock(2*ch_mul, ch_mul) # , seperable=False
        
        self.final=nn.Conv3d(ch_mul, out_chans, kernel_size=1)
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.max_pool(enc1))        
        enc3 = self.enc3(self.max_pool(enc2))        
        enc4 = self.enc4(self.max_pool(enc3))        
        
        middle = self.middle(self.max_pool(enc4))
        
        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1)
        
        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2)
        
        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 =self.dec3(up3)
        #print(enc1.size())
        #print(self.up4(dec3).size())
        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4)
        
        final=self.final(dec4)
        
        return final

class WNet(nn.Module):
    def __init__(self, k=1, ch_mul=64, in_chans=1, out_chans=1000):
        super(WNet, self).__init__()
        if out_chans==1000:
            out_chans=in_chans
        print('out_chans',k, ch_mul, in_chans, out_chans)
        self.UEnc=UEnc(k, ch_mul, in_chans).to(device)
        self.UDec=UDec(k, ch_mul, in_chans, out_chans).to(device)
    
    def forward(self, x, returns='both'):
        enc = self.UEnc(x)
        
        if returns=='enc':
            return enc
        
        dec=self.UDec(enc)
        
        if returns=='dec':
            return dec
        
        if returns=='both':
            return enc, dec
        
        else:
            raise ValueError('Invalid returns, returns must be in [enc dec both]')

