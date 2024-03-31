# import pandas as pd
# from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import argparse
# Data Exploration
# 1. The distribution of data instances (Train set / Test set)
# 2. The shape of each data instance (-> Shape after converting to numpy)


# Cleaning & Sampling
# 1. Split train set into train/validation
# 2. Potential data (Smote) / Undersampling, Oversampling

# Insights from Data Exploration
# Test comment
LABEL_MAP = {"fear":0, "disgust":1, "happy":2, "sad":3, "neutral":4, "angry":5, "surprise": 6}
EXPLORE = 1
DIM = 48

def convert_dataset(data_dir=None, split='train'):
    split_dir = os.path.join(data_dir, split)
    dataset = [] 
    labels  = []
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        for file in os.listdir(label_dir):
            img = Image.open(os.path.join(label_dir, file))
            img_array = np.array(img)
            dataset.append(img_array)
            labels.append(LABEL_MAP[label])
    return np.array(dataset), np.array(labels)
    
        
def data_exploration(split, labels):
    message = f"Split: {split})\n"
    for k, v in LABEL_MAP.items():
        num = sum([1 if elem==v else 0 for elem in labels])
        message += f" Labels : {k}, # of instances: {num}\n"
    print(message)


def main(args):
    # 1. We gonna load data which is JPG
    # 2. We should convert JPG -> array by using numpy 
    # 3. We gonna implement / import models (such as CNN) and just train with trainset / test on testset
    # Data Exploration
    data_dir = os.path.join(os.getcwd(), "data")
    print("Start loading the dataset.....")
    train_dataset, train_labels = convert_dataset(data_dir, split='train')
    test_dataset, test_labels  = convert_dataset(data_dir, split='test')
    assert len(train_dataset) == len(train_labels), "Not correct number, for train"
    assert len(test_dataset) == len(test_labels), "Not correct number, for test"
    print("Splitting into train / valset")
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(train_dataset, train_labels, test_size=0.25, random_state=42, stratify=train_labels)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # Explore data
    if EXPLORE:
        data_exploration('train', train_labels)
        data_exploration('val', val_labels)
        data_exploration('test', test_labels)
    
    # Just convert for a bit while for sampling
    train_dataset = train_dataset.reshape(train_dataset.shape[0], -1)
    
    if args.sample_method == 'oversample' or args.sample_method == 'all':
        ros= RandomOverSampler()
        train_dataset_os, train_labels_os = ros.fit_resample(train_dataset, train_labels)
        train_dataset_os = train_dataset_os.reshape(train_dataset_os.shape[0], DIM, DIM)

    if args.sample_method == 'undersample' or args.sample_method == 'all':
        rus= RandomUnderSampler()
        train_dataset_us, train_labels_us = rus.fit_resample(train_dataset, train_labels) 
        train_dataset_us = train_dataset_us.reshape(train_dataset_us.shape[0], DIM, DIM)
    
    if args.sample_method == 'smote' or args.sample_method == 'all':
        smote=SMOTE(random_state=42)
        train_dataset_smote, train_labels_smote = smote.fit_resample(train_dataset, train_labels)
        train_dataset_smote = train_dataset_smote.reshape(train_dataset_smote.shape[0], DIM, DIM)

    if EXPLORE:
        data_exploration("Oversampler", train_labels_os)
        data_exploration("Undersampler", train_labels_us)
        data_exploration("Smote", train_labels_smote)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default="all")
    
    args = parser.parse_args()
    main(args)

