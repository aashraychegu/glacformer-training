# Glacies Prime

## Introduction
The goal of Glacies Prime is to create a transformer based machine learning model that can separate a radio scope of a glacier into a sky, glacial bed (bed), and bedrock to aid in estimating the thickness of the glacial bed

## Requirements:
 - Python 3.8 or higher
 - pip install -r requirements.txt

## Organization
This directory is organized into  sections:
 - Preprocessing
 - Training
 - Data
 - Glacformer

### Preprocessing
The primary preprocessing script is preprocessing/preprocess.py
This script is used to turn raw data into a huggingface dataset for training the machine learning algorithm
See the Preprocessing folder for more information on preprocess.py

### Training
The primary script for training the machine learning algorithm is training/glacformer_training_script.py. See Training for more information on glacformer_training_script.py.

### Data
There are three directories that hold data
 - raw_data has two directories, csvs and tiffs. csvs holds the raw csvs with data from labelled scopes and tiffs of the labelled scopes
 - processed_images is an internal folder used for holding intermediate files from dataset generation
 - dataset is a folder holding previous versions of the dataset. The latest version of the dataset is automatically pushed to huggingface hub.

### Glacformer
This holds the internal model files during training. The model is then pushed to huggingface hub.


## Next Steps/Todo
 - Instead of saving a png file, save a .jfif file for lower file sizes