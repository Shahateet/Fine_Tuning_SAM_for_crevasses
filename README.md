# Fine_Tuning_SAM_for_crevasses

This repository is intended to provide a pipeline to fine tune the Segment-Anything model with fracture delineation to improve its capability of identifying the fractures.

We recommend spyder to visualize and edit the python codes.

## Installations

The required python packages are in requirements.txt. to install that use:

conda create --name FT_SAM --file requirements.txt

## Fine tuning SAM

The code to process the data is Fine_tune_SAM_fractures.py. It has 3 main parts: Data manipulation, image augmentation and training. The output is a .pth file with the trained weights.

## Inference
