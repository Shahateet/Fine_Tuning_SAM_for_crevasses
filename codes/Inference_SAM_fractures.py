#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:57:21 2024

@author: fernanka
"""
# %% Load required packages
#####################
# Required packages #
#####################
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify, unpatchify  #Only to handle large images
import random
from scipy import ndimage

from datasets import Dataset
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai

from transformers import SamProcessor
from transformers import SamModel

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

from transformers import SamModel, SamConfig, SamProcessor
import torch
from datasets import Dataset
from datasets import load_dataset
#####################

# %% Data manipulation

#Apply a trained model on large image
test_name = "//home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/Fine_Tuning_SAM_for_crevasses/Dataset/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_testset/stack.tif"
test_name_mask = "//home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/Fine_Tuning_SAM_for_crevasses/Dataset/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_testset/stack_mask.tif"

# proper stack the tif file
dataset = Image.open(test_name_mask)
h,w = np.shape(dataset)
tiffarray = np.zeros((h,w,dataset.n_frames))
for i in range(dataset.n_frames):
   dataset.seek(i)
   tiffarray[:,:,i] = np.array(dataset)
mask_ks_test = tiffarray.astype(np.double);
print(mask_ks_test.shape)
#
# proper stack the tif file
dataset = Image.open(test_name)
h,w = np.shape(dataset)
tiffarray = np.zeros((h,w,dataset.n_frames))
for i in range(dataset.n_frames):
   dataset.seek(i)
   tiffarray[:,:,i] = np.array(dataset)
img_ks_test = tiffarray.astype(np.double);
print(img_ks_test.shape)
#


img_ks_test_t=np.transpose(img_ks_test,axes=[2,0,1])
masks_ks_t=np.transpose(mask_ks_test,axes=[2,0,1])
#165 large images as tiff image stack


# Covert image size to 1024*1024
new_size = (1024, 1024)
img_ks_test_t_pil_res=np.zeros((img_ks_test_t.shape[0],new_size[0],new_size[1]))
for i in range(img_ks_test_t.shape[0]):
    img_ks_test_t_pil = Image.fromarray(img_ks_test_t[i])
# Resize the image
    img_ks_test_t_pil_res[i] = img_ks_test_t_pil.resize(new_size, Image.Resampling.LANCZOS)
img_ks_test_t_pil_res_array=np.array(img_ks_test_t_pil_res)
#############
# Covert mask size to 1024*1024
new_size = (1024, 1024)
masks_ks_t_pil_res=np.zeros((masks_ks_t.shape[0],new_size[0],new_size[1]))
for i in range(img_ks_test_t.shape[0]):
    masks_ks_t_pil = Image.fromarray(masks_ks_t[i])
# Resize the image
    masks_ks_t_pil_res[i] = masks_ks_t_pil.resize(new_size, Image.Resampling.LANCZOS)
masks_ks_t_pil_res_array=np.array(masks_ks_t_pil_res)
#############

large_images_test = img_ks_test_t_pil_res_array
large_masks_test = masks_ks_t_pil_res_array
large_images_test.shape
large_masks_test.shape
# Now. lets divide these large images into smaller patches for training. We can use patchify or write custom code.
#Desired patch size for smaller images and step size.

patch_size = 256
step = 256

all_img_patches = [] #empty tensor
for img in range(large_images_test.shape[0]): # iterate through images
    large_image = large_images_test[img] # select individual images
    patches_img = patchify(large_image, (patch_size, patch_size), step=step)  # create new patches of this image #Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]): # iterate through the individual patches with next line
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img / 255.).astype(np.uint8)
            all_img_patches.append(single_patch_img) # stack patches
            
images = np.array(all_img_patches) # transform the stacked patches into a numpy array

#Lets do the same for masks
all_mask_patches = []
for img in range(large_masks_test.shape[0]):
    large_mask = large_masks_test[img]
    patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i,j,:,:]
            single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)

images.shape
masks.shape


# Now, lets delete empty masks as they may cause issues later on during training. If a batch contains empty masks then the loss function will throw an error as it may not know how to handle empty tensors.
# Create a list to store the indices of non-empty masks
valid_indices = [i for i, images in enumerate(images) if images.max() != 0] # take valid indices
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images_test = images[valid_indices]
filtered_masks_test = masks[valid_indices]
print("Image shape:", filtered_images_test.shape)  # e.g., (num_frames, height, width, num_channels)
print("Mask shape:", filtered_images_test.shape)

"""
input_points (torch.FloatTensor of shape (batch_size, num_points, 2)) —
Input 2D spatial points, this is used by the prompt encoder to encode the prompt.
Generally yields to much better results. The points can be obtained by passing a
list of list of list to the processor that will create corresponding torch tensors
of dimension 4. The first dimension is the image batch size, the second dimension
is the point batch size (i.e. how many segmentation masks do we want the model to
predict per input point), the third dimension is the number of points per segmentation
mask (it is possible to pass multiple points for a single mask), and the last dimension
is the x (vertical) and y (horizontal) coordinates of the point. If a different number
of points is passed either for each image, or for each mask, the processor will create
“PAD” points that will correspond to the (0, 0) coordinate, and the computation of the
embedding will be skipped for these points using the labels.

"""
# Define the size of your array
array_size = 256

# Define the size of your grid
grid_size = 10

# Generate the grid points
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)

# Generate a grid of coordinates
xv, yv = np.meshgrid(x, y)

# Convert the numpy arrays to lists
xv_list = xv.tolist()
yv_list = yv.tolist()

# Combine the x and y coordinates into a list of list of lists
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

#We need to reshape our nxn grid to the expected shape of the input_points tensor
# (batch_size, point_batch_size, num_points_per_image, 2),
# where the last dimension of 2 represents the x and y coordinates of each point.
#batch_size: The number of images you're processing at once.
#point_batch_size: The number of point sets you have for each image.
#num_points_per_image: The number of points in each set.
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
     

print(np.array(input_points).shape)

filtered_images_test.shape

# Select a random patch for segmentation

# Compute the total number of 256x256 arrays
#num_arrays = patches.shape[0] * patches.shape[1]
# Select a random index
#index = np.random.choice(num_arrays)
# Compute the indices in the original array
#i = index // patches.shape[1]
#j = index % patches.shape[1]


# prepare image for the model

# AM expects RGB images. So:
dataset_test = {
    "image": [Image.fromarray(img.squeeze(), mode='L').convert("RGB") for img in filtered_images_test],
    "label": [Image.fromarray(mask.squeeze(), mode='L').convert("I") for mask in filtered_masks_test],
}


# %% Inference

idx = random.randint(0,len(dataset_test["image"])-1)
#KS idx=11 # a good example

single_patch=dataset_test['image'][idx] # 34,26 are good and 10 show how bad

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_fracture_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_fracture_model.load_state_dict(torch.load("//home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/Fine_Tuning_SAM_for_crevasses/weights/fracture_model_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_fracture_model.to(device)

#First try without providing any prompt (no bounding box or input_points)
# inputs = processor(single_patch, return_tensors="pt")
#Now try with bounding boxes. Remember to uncomment.
inputs = processor(single_patch, input_points=input_points, return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}
my_fracture_model.eval()


# forward pass
with torch.no_grad():
  outputs = my_fracture_model(**inputs, multimask_output=False)

# apply sigmoid
single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)


# % Plot example of the test result and groundtruth
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the left
axes[0,0].imshow(np.array(dataset_test["image"][idx]), cmap='gray')  # Assuming the first image is grayscale
axes[0,0].set_title("Optical Image")
 
# Plot the second image on the right
a=np.array(dataset_test['label'][idx])*255
axes[0,1].imshow(a, cmap='gray')  # Assuming the second image is grayscale
axes[0,1].set_title("Mask groundtruth")

# Plot the second image on the right
axes[1,0].imshow(single_patch_prob)  # Assuming the second image is grayscale
axes[1,0].set_title("Probability Map")

# Plot the second image on the right
axes[1,1].imshow(single_patch_prediction, cmap='gray')  # Assuming the second image is grayscale
axes[1,1].set_title("Mask predicted")

plt.show()

# %% Display the images side by side
"""
plt.show()

fig, axes = plt.subplots()

axes.imshow(np.array(single_patch), cmap="gray")
"""

