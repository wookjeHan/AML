# import pandas as pd
# from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score as f1_score_sklearn
from torcheval.metrics.functional import multiclass_f1_score
import argparse

from models import Resnet, VGG
from models import SVC_classifier
from models import KNN_classifier
# Data Exploration
# 1. The distribution of data instances (Train set / Test set)
# 2. The shape of each data instance (-> Shape after converting to numpy)


# Cleaning & Sampling
# 1. Split train set into train/validation
# 2. Potential data (Smote) / Undersampling, Oversampling

# Insights from Data Exploration
# Test comment
LABEL_MAP = {"fear":0, "disgust":1, "happy":2, "sad":3, "neutral":4, "angry":5, "surprise": 6}
EXPLORE = 0
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
    
    train_dataset = train_dataset.reshape(train_dataset.shape[0], DIM, DIM)
    
    if args.model == "Resnet":
        #Test Code for Resnet
        resnet = Resnet()
        resnet.train(train_dataset, train_labels, val_dataset, val_labels)
        predictions = resnet.predict(test_dataset)
        print(f"LEN PREDICTIONS : {len(predictions)}")
        print(predictions[:10])
        test_accuracy = np.mean(predictions == test_labels)
        print("ACC : ", test_accuracy)
        print(np.array(test_labels).size, np.array(predictions).size)
        f1_score = multiclass_f1_score(torch.tensor(test_labels), torch.tensor(predictions), num_classes=7, average="weighted")
        print(f"Test Accuracy: {test_accuracy}, f1_score: {f1_score}")
        
    if args.model == "SVC_classifier":
        svc = SVC_classifier()

        # must reshape sampling methods for SVC to work
        train_x_smote = train_dataset_smote.reshape(train_dataset_smote.shape[0], -1)
        train_x_us = train_dataset_us.reshape(train_dataset_us.shape[0], -1)
        train_x_os = train_dataset_os.reshape(train_dataset_os.shape[0], -1)
        val_x_reshaped = val_dataset.reshape(val_dataset.shape[0], -1)
        test_x_reshaped = test_dataset.reshape(test_dataset.shape[0],-1)

        # choose original df or a sampling method to train on
        svc.train(train_x_us, train_labels_us, val_x_reshaped, val_labels)
        predictions = svc.predict(test_x_reshaped)
        test_accuracy = np.mean(predictions == test_labels)
        f1 = f1_score_sklearn(test_labels, predictions, average='weighted')
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Weighted F1 Score: {f1}")
        
    if args.model == "KNN":
       # Create a dictionary to hold the different sampling datasets
       sampling_methods = {
          'original': (train_dataset, train_labels),
          'OverSampling': (train_dataset_os, train_labels_os),
          'UnderSampling': (train_dataset_us, train_labels_us),
          'SMOTE': (train_dataset_smote, train_labels_smote)
        }

       # Create a StandardScaler instance for data normalization
       scaler = StandardScaler()

       # Loop over each sampling method
       for method, (data, labels) in sampling_methods.items():
          print(f"Training KNN model with {method} enhanced data...")
          # Reshape and scale the data
          data_reshaped = data.reshape(data.shape[0], -1)
          data_scaled = scaler.fit_transform(data_reshaped)

          # Initialize the KNN model
          knn = KNN_classifier(n_neighbors=args.k)
         
          # Train the KNN model
          knn.train(data_scaled, labels)

          # Prepare the test data and make predictions
          test_data_reshaped = test_dataset.reshape(test_dataset.shape[0], -1)
          test_data_scaled = scaler.transform(test_data_reshaped)
          print("Predicting with KNN model...")
          predictions = knn.predict(test_data_scaled)

          # Calculate accuracy and F1 score        
          accuracy, f1 = knn.evaluate(test_data_scaled, test_labels)
          print(f"Test Accuracy for KNN with {method}: {accuracy:.2f}")
          print(f"Test F1 Score for KNN with {method}: {f1:.2f}")
           
    # Test code for VGG-16
    if args.model == "VGG":
        vgg = VGG()
        test_loader = vgg._create_dataloader(test_dataset, test_labels, batch_size=32)
        sampling_methods = {
            'original': (train_dataset, train_labels),
            'OverSampling': (train_dataset_os, train_labels_os),
            'UnderSampling': (train_dataset_us, train_labels_us),
            'SMOTE' : (train_dataset_smote, train_labels_smote)
        }
        for method, (data, labels) in sampling_methods.items():
            print(f"Training VGG model with {method} enhanced data...")
            vgg.train(data, labels, val_dataset, val_labels)
            print("Predicting with VGG model...")
            predictions = vgg.predict(test_loader)
            test_accuracy = np.mean(predictions == test_labels)
            f1_score = multiclass_f1_score(torch.tensor(test_labels), torch.tensor(predictions), num_classes=7, average="weighted")
            print(f"Test Accuracy for VGG: {test_accuracy:.2f}, F1: {f1_score}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default="all")
    parser.add_argument('--model', type=str, default='SVC')    
    #parser.add_argument('--model', type=str, default="SVC_classifier")
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for the KNN model.')
    args = parser.parse_args()
    main(args)

