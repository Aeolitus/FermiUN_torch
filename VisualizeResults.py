#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser("TorchUnet Evaluation")
parser.add_argument("runname", type=str)
parser.add_argument("modelpath", type=str)
args = parser.parse_args()

IMAGE_FOLDER = ['../FermiUNData/images_0', '../FermiUNData/images_1',
                '../FermiUNData/images_2', '../FermiUNData/images_3',
                '../FermiUNData/images_4']
OUTPUT_DIR = '../Results/' + args.runname
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE = 444
MASK_RADIUS = 120
TEST_SPLIT = .15
VALIDATION_SPLIT = .15
BATCH_SIZE = 32


class EncoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, channels=(1, 32, 64, 128, 256, 512)):
        super().__init__()
        self.EncoderBlocks = nn.ModuleList(
            [EncoderBlock(channels[i], channels[i+1])
             for i in range(len(channels)-1)])
        self.MaxPool = nn.MaxPool2d(2)

    def forward(self, x):
        outputs = []
        for block in self.EncoderBlocks:
            x = block(x)
            outputs.append(x)
            x = self.MaxPool(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, channels=(512, 256, 128, 64, 32)):
        super().__init__()
        self.channels = channels
        self.UpConvolutions = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2)
             for i in range(len(channels)-1)])
        self.DecoderBlocks = nn.ModuleList(
            [EncoderBlock(channels[i], channels[i+1])
             for i in range(len(channels)-1)])

    def crop(self, encoder_features, x):
        # crop the encoder features to match the current dimensions
        _, _, height, width = x.shape
        encoder_features = CenterCrop([height, width])(encoder_features)
        return encoder_features

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.UpConvolutions[i](x)
            cropped_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, cropped_feature], dim=1)
            x = self.DecoderBlocks[i](x)
        return x


class Head(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.height = height
        self.width = width
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = CenterCrop([self.height, self.width])(x)
        return x


class Unet(nn.Module):
    def __init__(self, encoder_channels=(1, 32, 64, 128, 256, 512), decoder_channels=(512, 256, 128, 64, 32)):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = Head(2*MASK_RADIUS, 2*MASK_RADIUS, decoder_channels[-1])

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(decoder_features)
        return out


# Lets write our custom datasets, dataloaders and related functions first
class FermiUNDataset(Dataset):
    def __init__(self, image_dirs, transform=None, target_transform=None):
        self.image_dirs = image_dirs
        self.image_list = []
        for folder in image_dirs:
            _elements = os.listdir(folder)
            _elements = [os.path.join(folder, el) for el in _elements]
            self.image_list += _elements
        self.transform = transform
        self.target_transform = target_transform

        # Make the mask for our images - circular with radius MASK_RADIUS
        _scale = np.arange(IMAGE_SIZE)
        _mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        _mask[(_scale[np.newaxis, :] - (IMAGE_SIZE - 1) / 2) ** 2
              + (_scale[:, np.newaxis] - (IMAGE_SIZE - 1) / 2) ** 2
              > MASK_RADIUS ** 2] = 1
        self.mask = torch.from_numpy(_mask)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _image = np.load(self.image_list[index])
        _image = _image[np.newaxis, ...]
        _image = torch.from_numpy(_image)
        _masked = _image * self.mask
        _target = CenterCrop([2*MASK_RADIUS, 2*MASK_RADIUS])(_image)
        if self.transform:
            _masked = self.transform(_masked)
        if self.target_transform:
            _target = self.target_transform(_target)
        return _masked, _target


# The dataset we draw all images from
FermiUNData = FermiUNDataset(IMAGE_FOLDER)

# Split into training, validation, test
dataset_size = len(FermiUNData)
dataset_indices = list(range(dataset_size))
np.random.seed(42)
np.random.shuffle(dataset_indices)
first_split, second_split = (int(np.floor(VALIDATION_SPLIT*dataset_size)),
                             dataset_size-int(np.floor(TEST_SPLIT*dataset_size)))
_, _, test_indices = (dataset_indices[:first_split], dataset_indices[first_split:second_split],
                      dataset_indices[second_split:])
FermiUNLoaderTesting = DataLoader(FermiUNData, batch_size=BATCH_SIZE,
                                  sampler=SubsetRandomSampler(test_indices), num_workers=4, pin_memory=True)

model = torch.load(args.modelpath).to("cuda", non_blocking=True)
model.eval()

for batch, (X, y) in enumerate(FermiUNLoaderTesting):
    X, y = X.float().to("cuda", non_blocking=True), y.float().to("cuda", non_blocking=True)
    pred = model(X)
    for i in range(10):
        fig = plt.figure()
        im1_gt = np.squeeze(y[i, :, :].cpu().detach().numpy())
        im1_pr = np.squeeze(y[i,:,:].cpu().detach().numpy()-pred[i, :, :].cpu().detach().numpy())
        im1_df = im1_pr-im1_gt
        fig.add_subplot(1, 3, 1)
        plt.title("Ground Truth")
        plt.axis("off")
        im1 = plt.imshow(im1_gt)
        vmin, vmax = im1.get_clim()
        fig.add_subplot(1, 3, 2)
        plt.title("Prediction")
        plt.axis("off")
        plt.imshow(im1_pr, vmin=vmin, vmax=vmax)
        fig.add_subplot(1, 3, 3)
        plt.title("Difference")
        plt.axis("off")
        plt.imshow(im1_df, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join(OUTPUT_DIR, 'Image_' + str(i) + '.png'))




