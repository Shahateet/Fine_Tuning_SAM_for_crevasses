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


# Load tiff stack images and masks

in_name_mask="/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_trainset/stack_mask.tif"
in_name_or="/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_trainset/stack.tif"

# proper stack the tif file
dataset = Image.open(in_name_mask)
h,w = np.shape(dataset)
tiffarray = np.zeros((h,w,dataset.n_frames))
for i in range(dataset.n_frames):
   dataset.seek(i)
   tiffarray[:,:,i] = np.array(dataset)
mask_ks_test = tiffarray.astype(np.double);
print(mask_ks_test.shape)
#
# proper stack the tif file
dataset = Image.open(in_name_or)
h,w = np.shape(dataset)
tiffarray = np.zeros((h,w,dataset.n_frames))
for i in range(dataset.n_frames):
   dataset.seek(i)
   tiffarray[:,:,i] = np.array(dataset)
img_ks = tiffarray.astype(np.double);
print(img_ks.shape)
#


img_ks_t=np.transpose(img_ks,axes=[2,0,1])
masks_ks_t=np.transpose(mask_ks_test,axes=[2,0,1])
#165 large images as tiff image stack


# Covert image size to 1024*1024
new_size = (1024, 1024)
img_ks_t_pil_res=np.zeros((img_ks_t.shape[0],new_size[0],new_size[1]))
for i in range(img_ks_t.shape[0]):
    img_ks_t_pil = Image.fromarray(img_ks_t[i])
# Resize the image
    img_ks_t_pil_res[i] = img_ks_t_pil.resize(new_size, Image.Resampling.LANCZOS)
img_ks_t_pil_res_array=np.array(img_ks_t_pil_res)
#############
# Covert mask size to 1024*1024
new_size = (1024, 1024)
masks_ks_t_pil_res=np.zeros((masks_ks_t.shape[0],new_size[0],new_size[1]))
for i in range(img_ks_t.shape[0]):
    masks_ks_t_pil = Image.fromarray(masks_ks_t[i])
# Resize the image
    masks_ks_t_pil_res[i] = masks_ks_t_pil.resize(new_size, Image.Resampling.LANCZOS)
masks_ks_t_pil_res_array=np.array(masks_ks_t_pil_res)
#############

large_images = img_ks_t_pil_res_array
large_masks = masks_ks_t_pil_res_array
large_images.shape
large_masks.shape
# Now. lets divide these large images into smaller patches for training. We can use patchify or write custom code.
#Desired patch size for smaller images and step size.
patch_size = 256
step = 256

all_img_patches = [] #empty tensor
for img in range(large_images.shape[0]): # iterate through images
    large_image = large_images[img] # select individual images
    patches_img = patchify(large_image, (patch_size, patch_size), step=step)  # create new patches of this image #Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]): # iterate through the individual patches with next line
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img / 255.).astype(np.uint8)
            all_img_patches.append(single_patch_img) # stack patches
            
images = np.array(all_img_patches) # transform the stacked patches into a numpy array

#Lets do the same for masks
all_mask_patches = []
for img in range(large_masks.shape[0]):
    large_mask = large_masks[img]
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
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0] # take valid indices
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
print("Mask shape:", filtered_masks.shape)

# Lets create a 'dataset' that serves us input images and masks for the rest of our journey.
# Convert the NumPy arrays to Pillow images and store them in a dictionary

# Add a channel dimension to images and masks
filtered_images = filtered_images[..., np.newaxis]  # Adds a channel dimension (height, width) -> (height, width, 1)
#a.astype(filtered_images.int64)
filtered_masks = filtered_masks[..., np.newaxis]    # Adds a channel dimension (height, width) -> (height, width, 1)


dataset_dict = {
    "image": [Image.fromarray(img.squeeze(), mode='L').convert("RGB") for img in filtered_images],
    "label": [Image.fromarray(mask.squeeze(), mode='L').convert("I") for mask in filtered_masks],
}
dataset = Dataset.from_dict(dataset_dict)
dataset
# We can visualize an example:
example = dataset[1]
image = example["image"]
image

# %% Test dataset image augmentation

# Image Classification
import torch
from torchvision.transforms import v2
import torchvision
import torchvision.transforms.functional as transform
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img_pir = torchvision.io.read_image("/home/fernanka/Downloads/piratinha.jpeg")

# Define the transformations
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            return transform.hflip(img), transform.hflip(mask)
        return img, mask
    
class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            return transform.vflip(img), transform.vflip(mask)
        return img, mask
    
class GaussianBlurPair:
    def __init__(self, kernel_size, sigma):
        self.transform = v2.GaussianBlur(kernel_size, sigma)

    def __call__(self, img, mask):
        return self.transform(img), self.transform(mask)
    
class RandomResizedCropPair:
    def __init__(self, size, antialias=True):
        self.size = size
        self.antialias = antialias

    def __call__(self, img, mask):
        params = v2.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
        img = transform.resized_crop(img, *params, self.size, interpolation=transform.InterpolationMode.BILINEAR)
        mask = transform.resized_crop(mask, *params, self.size, interpolation=transform.InterpolationMode.NEAREST)
        return img, mask
    
class RandomRotationPair:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, mask):
        img = transform.rotate(img, self.angle)
        mask = transform.rotate(mask, self.angle)
        return img, mask


# Compose transformations
transforms = v2.Compose([
    RandomVerticalFlipPair(p=0.5),
    RandomHorizontalFlipPair(p=0.5),
    GaussianBlurPair(1, 5),
    # RandomResizedCropPair(size=(224, 224)),
    RandomRotationPair(random.randint(-180, 180))
])

pil_image_trans = dataset[:]["image"]
pil_label_trans = dataset[:]["label"]


for i in range(len(dataset[:]["image"])):    
    img=dataset[i]["image"]
    tensor_image = transform.to_tensor(img)
    
    label=dataset[i]["label"]
    tensor_label = transform.to_tensor(label)
    
    tensor_image_trans, tensor_label_trans = transforms(tensor_image, tensor_label)
    pil_image_trans[i] = transform.to_pil_image(tensor_image_trans)
    pil_label_trans[i] = transform.to_pil_image(tensor_label_trans)
    
dataset_transformed_dict = {
    "image": [img.convert("RGB") for img in pil_image_trans],
    "label": [mask.convert("I") for mask in pil_label_trans],
}

# We can visualize an example:
dataset_transformed = Dataset.from_dict(dataset_transformed_dict)
idx = random.randint(0,len(dataset_transformed["image"])-1)
example_trans = dataset_transformed[idx]
image_trans = example_trans["image"]
label_trans = example_trans["label"]

# We can visualize an example:
example = dataset[idx]
image_or = example["image"]
mask_or = example["label"]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the left
axes[0,0].imshow(image_or, cmap='gray')  # Assuming the first image is grayscale
axes[0,0].set_title("Original Image")
 
# Plot the second image on the right
a_or=np.array(mask_or)*255
axes[0,1].imshow(a_or, cmap='gray')  # Assuming the second image is grayscale
axes[0,1].set_title("Original Mask")

# Plot the first image on the left
axes[1,0].imshow(image_trans, cmap='gray')  # Assuming the first image is grayscale
axes[1,0].set_title("Transformed Image")
 
# Plot the second image on the right
a_trans=np.array(label_trans)*255
axes[1,1].imshow(a_trans, cmap='gray')  # Assuming the second image is grayscale
axes[1,1].set_title("Transformed Mask")

plt.show()

# Concatenate the datasets
big_dataset_dict = {
    "image": dataset_dict["image"] + dataset_transformed_dict["image"],
    "label": dataset_dict["label"] + dataset_transformed_dict["label"],
}

# Print the lengths to verify concatenation
print(f"Total images: {len(big_dataset_dict['image'])}")
print(f"Total labels: {len(big_dataset_dict['label'])}")

dataset = Dataset.from_dict(big_dataset_dict)
dataset

# =============================================================================
# # %% Test dataset image augmentation Pirate
# 
# # Image Classification
# import torch
# from torchvision.transforms import v2
# import torchvision
# import torchvision.transforms.functional as transform
# import matplotlib.pyplot as plt
# import numpy as np
# 
# # Read the image
# img_pir = torchvision.io.read_image("/home/fernanka/Downloads/piratinha.jpeg")
# 
# # Define the transformations
# class RandomHorizontalFlipPair:
#     def __init__(self, p=0.5):
#         self.p = p
# 
#     def __call__(self, img, mask):
#         if torch.rand(1).item() < self.p:
#             return transform.hflip(img), transform.hflip(mask)
#         return img, mask
#     
# class RandomVerticalFlipPair:
#     def __init__(self, p=0.5):
#         self.p = p
# 
#     def __call__(self, img, mask):
#         if torch.rand(1).item() < self.p:
#             return transform.vflip(img), transform.vflip(mask)
#         return img, mask
#     
# class GaussianBlurPair:
#     def __init__(self, kernel_size, sigma):
#         self.transform = v2.GaussianBlur(kernel_size, sigma)
# 
#     def __call__(self, img, mask):
#         return self.transform(img), self.transform(mask)
#     
# class RandomResizedCropPair:
#     def __init__(self, size, antialias=True):
#         self.size = size
#         self.antialias = antialias
# 
#     def __call__(self, img, mask):
#         params = v2.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
#         img = transform.resized_crop(img, *params, self.size, interpolation=transform.InterpolationMode.BILINEAR)
#         mask = transform.resized_crop(mask, *params, self.size, interpolation=transform.InterpolationMode.NEAREST)
#         return img, mask
#     
# class RandomRotationPair:
#     def __init__(self, angle):
#         self.angle = angle
# 
#     def __call__(self, img, mask):
#         img = transform.rotate(img, self.angle)
#         mask = transform.rotate(mask, self.angle)
#         return img, mask
# 
# 
# # Compose transformations
# transforms = v2.Compose([
#     RandomVerticalFlipPair(p=0.5),
#     RandomHorizontalFlipPair(p=0.5),
#     GaussianBlurPair(1, 5),
#     RandomResizedCropPair(size=(224, 224)),
#     RandomRotationPair(random.randint(-180, 180))
# ])
# 
# 
# # Apply transformations to image and mask
# tensor_image_trans, tensor_label_trans = transforms(img_pir, img_pir)
# image_trans = transform.to_pil_image(tensor_image_trans)
# label_trans = transform.to_pil_image(tensor_label_trans)
# 
# # Original image and mask
# image_or = transform.to_pil_image(img_pir)
# mask_or = transform.to_pil_image(img_pir)
# 
# # Visualize
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# 
# # Plot the original image on the left
# axes[0,0].imshow(image_or)
# axes[0,0].set_title("Original Image")
# 
# # Plot the original mask on the right
# a_or = np.array(mask_or) * 255
# axes[0,1].imshow(a_or)
# axes[0,1].set_title("Original Mask")
# 
# # Plot the transformed image on the left
# axes[1,0].imshow(image_trans)
# axes[1,0].set_title("Transformed Image")
# 
# # Plot the transformed mask on the right
# a_trans = np.array(label_trans) * 255
# axes[1,1].imshow(a_trans)
# axes[1,1].set_title("Transformed Mask")
# 
# plt.show()
# =============================================================================



# %% Show mask and bbox function

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

fig, axes = plt.subplots()

axes.imshow(np.array(image))
ground_truth_seg = np.array(example["label"])
show_mask(ground_truth_seg, axes)
axes.title.set_text(f"Ground truth mask")
axes.axis("off")

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

# %% Training cell

from torch.utils.data import Dataset

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)
  
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)
  
batch["ground_truth_mask"].shape

model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
    
# train the model

# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    
# Important note here: as we used the Dice loss with sigmoid=True, we need to make sure to appropriately apply a sigmoid activation function to the predicted masks. Hence we won't use the processor's post_process_masks method here.
#unpatchify(np.asarray(dataset["image"]), (3,3))
# let's take a random training example
idx = random.randint(0,len(dataset["image"])-1)

medsam_seg_prob_array = np.zeros(np.array(dataset["label"]).shape)
medsam_seg_array = np.zeros(np.array(dataset["label"]).shape)
for idx in range(len(dataset["image"])): # iterate all the training images
# load image
    image = dataset[idx]["image"]
    image
# get box prompt based on ground truth segmentation map
    ground_truth_mask = np.array(dataset[idx]["label"])
    prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    for k,v in inputs.items():
        print(k,v.shape)     
  
    model.eval()

# forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
  
# apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    
    medsam_seg_prob_array[idx]=medsam_seg_prob
    medsam_seg_array[idx]=medsam_seg

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
# Now, lets load a new image and segment it using our trained model. NOte that we need to provide some prompt. Since we do not know where the objects are going to be we cannot supply bounding boxes. So let us provide a grid of points as our prompt.

pred_sam_full_image=np.zeros(masks.shape)
prob_pred_sam_full_image=np.zeros(masks.shape)
for i in range(len(valid_indices)):
    pred_sam_full_image[valid_indices[i]]=medsam_seg_array[i]
    prob_pred_sam_full_image[valid_indices[i]]=medsam_seg_prob_array[i]
    
# unpatchify(pred_sam_full_image, (1024,1024))
# =============================================================================
# KS Unpatchifying the data
# =============================================================================
unpatch_masks=np.zeros(large_images.shape)
unpatch_masks_prob=np.zeros(large_images.shape)
patch_per_col=int(large_masks.shape[2]/pred_sam_full_image.shape[2])
patch_per_row=int(large_masks.shape[1]/pred_sam_full_image.shape[1])

for i in range(large_masks.shape[0]):
    for j in range(patch_per_row):
        line_image=np.concatenate((pred_sam_full_image[(i*patch_per_col*patch_per_row)+j*patch_per_col:(i*patch_per_col*patch_per_row)+j*patch_per_col+patch_per_col]),axis=1)
        unpatch_masks[i,j*patch_size:j*patch_size+patch_size,0:large_images.shape[1]]=line_image
        
        line_image=np.concatenate((prob_pred_sam_full_image[(i*patch_per_col*patch_per_row)+j*patch_per_col:(i*patch_per_col*patch_per_row)+j*patch_per_col+patch_per_col]),axis=1)
        unpatch_masks_prob[i,j*patch_size:j*patch_size+patch_size,0:large_images.shape[1]]=line_image

# prob_pred_sam_reshape=np.reshape(images, large_images.shape)

fig, axes = plt.subplots(1, 2, figsize=(5, 10))

axes[0].imshow(np.array(large_images[0]), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Original Image")

axes[1].imshow(unpatch_masks[0], cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Unpatchfied")
plt.show()

idx = random.randint(0,unpatch_masks.shape[0]-1)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the left
axes[0,0].imshow(np.array(large_images[idx]), cmap='gray')  # Assuming the first image is grayscale
axes[0,0].set_title("Optical Image")
 
# Plot the second image on the right
a=np.array(large_masks[idx])*255
axes[0,1].imshow(a, cmap='gray')  # Assuming the second image is grayscale
axes[0,1].set_title("Mask groundtruth")

# Plot the second image on the right
axes[1,0].imshow(unpatch_masks_prob[idx])  # Assuming the second image is grayscale
axes[1,0].set_title("Probability Map")

# Plot the second image on the right
axes[1,1].imshow(unpatch_masks[idx], cmap='gray')  # Assuming the second image is grayscale
axes[1,1].set_title("Mask predicted")

plt.show()

# Save the model's state dictionary to a file
torch.save(model.state_dict(), "/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/fracture_model_checkpoint.pth")
# =============================================================================
# %% Plot example of the training result and groundtruth
    
idx = random.randint(0,len(dataset["image"])-1)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the left
axes[0,0].imshow(np.array(dataset["image"][idx]), cmap='gray')  # Assuming the first image is grayscale
axes[0,0].set_title("Optical Image")
 
# Plot the second image on the right
a=np.array(dataset[idx]['label'])*255
axes[0,1].imshow(a, cmap='gray')  # Assuming the second image is grayscale
axes[0,1].set_title("Mask groundtruth")

# Plot the second image on the right
axes[1,0].imshow(prob_pred_sam_full_image[valid_indices[idx]])  # Assuming the second image is grayscale
axes[1,0].set_title("Probability Map")

# Plot the second image on the right
axes[1,1].imshow(pred_sam_full_image[valid_indices[idx]], cmap='gray')  # Assuming the second image is grayscale
axes[1,1].set_title("Mask predicted")

plt.show()

# %% Fake inference: Made on trained data
# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_fracture_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_fracture_model.load_state_dict(torch.load("/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/fracture_model_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_fracture_model.to(device)

# let's take a random training example
idx = random.randint(0, filtered_images.shape[0]-1)
idx=60

# load image
test_image = dataset[idx]["image"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

my_fracture_model.eval()

# forward pass
with torch.no_grad():
    outputs = my_fracture_model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the first image on the left
axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(medsam_seg, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Plot the second image on the right
axes[2].imshow(medsam_seg_prob)  # Assuming the second image is grayscale
axes[2].set_title("Probability Map")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %% Test inference

#Apply a trained model on large image
test_name = "/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_testset/stack.tif"
test_name_mask = "/home/fernanka/Desktop/IGE-CryoDyn/IceDaM/Projects/FT-SAM_fractures/ZHAO_2023_DATA_Antarctic-fracture-detection-master/data_testset/stack_mask.tif"

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

idx = random.randint(0,len(dataset_test["image"])-1)
#KS idx=11 # a good example

single_patch=dataset_test['image'][idx]

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

