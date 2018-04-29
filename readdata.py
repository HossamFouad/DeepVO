# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:09:29 2018

@author: HOSSAM ABDELHAMID
"""

import scipy.misc
import csv
import cv2
import numpy as np
from scipy.io import loadmat
x = loadmat('perm.mat')
# modalities 
batch_size=32
time_stamp=16
class DataReader(object):
    def __init__(self):
        self.load()

    def load(self):
        self.pose_train = []     #input data
        self.pose=[]
        self.image_train = []
        self.train_batch_pointer = 0 #pointer for taking mini batch one after another
        self.test_batch_pointer = 0
        self.num_images = 0  # Number of training samples
        # CVS file that has all images names, Class and quality factor
        with open('training_data.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Data...")
            for row in reader:
                self.image_train.append(row['Nr']+"/"+row['sequence']+row['image']+'.png')
                self.pose=[row['x'],row['y'],row['z'],row['theta_x'],row['theta_y'],row['theta_z']]
                self.pose_train.append(self.pose)
                self.num_images += 1       
        print('Total training data: ' + str(self.num_images))
        
        self.pose_test = []     #input data
        self.poset=[]
        self.image_test = []
        self.total_test = 0
        # CVS file that has all images names, Class and quality factor
        with open('TEST.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Testing Data ...")
            for row in reader:
                self.image_test.append(row['Nr']+"/"+row['sequence']+row['image']+'.png')
                self.poset=[row['x'],row['y'],row['z'],row['theta_x'],row['theta_y'],row['theta_z']]
                self.pose_test.append(self.poset)
                self.total_test += 1
        print('Total test data: ' + str(self.total_test))

        # Get Random mini batch of size batch_size
    def load_train_batch(self, batch_size):
        img_batch = []
        label_batch = []
        for i in range(0, batch_size):
            img_stacked_series = []
            labels_series = []                 
            initial=x['y'][0][(self.train_batch_pointer + i) % self.num_images]
            ref=self.pose_train[initial,:]
            for j in range(initial, initial+time_stamp): 
       # Normalizing and Subtracting mean intensity value of the corresponding image
                img1 = scipy.misc.imread(self.image_train[j])
                img1 = img1/np.max(img1)
                img1 = img1 - np.mean(img1)
                img1 = cv2.resize(img1, (1280,384), fx=0, fy=0)
                img2 = scipy.misc.imread(self.image_train[j+1])
                img2 = img2/np.max(img2)
                img2 = img2 - np.mean(img2)
                img2 = cv2.resize(img2, (1280,384), fx=0, fy=0)
                img_aug = np.concatenate([img1, img2], -1)
                img_stacked_series.append(img_aug)
                pose = self.pose_train[j,:] - ref
                labels_series.append(pose)
            img_batch.append(img_stacked_series)
            label_batch.append(labels_series)
        label_batch = np.array(label_batch)
        img_batch = np.array(img_batch)
        self.train_batch_pointer+=batch_size
        return img_batch, label_batch
    
    def load_test_data(self, test_size):
        img_batch = []
        label_batch = []
        for i in range(0, test_size):
            img_stacked_series = []
            labels_series = []  
            x=list(range(self.total_test))               
            initial=x[(self.test_batch_pointer + i)]
            ref=self.pose_train[initial,:]
            for j in range(initial, initial+time_stamp): 
       # Normalizing and Subtracting mean intensity value of the corresponding image
                img1 = scipy.misc.imread(self.image_test[j])
                img1 = img1/np.max(img1)
                img1 = img1 - np.mean(img1)
                img1 = cv2.resize(img1, (1280,384), fx=0, fy=0)
                img2 = scipy.misc.imread(self.image_test[j+1])
                img2 = img2/np.max(img2)
                img2 = img2 - np.mean(img2)
                img2 = cv2.resize(img2, (1280,384), fx=0, fy=0)
                img_aug = np.concatenate([img1, img2], -1)
                img_stacked_series.append(img_aug)
                pose = self.pose_test[j,:] - ref
                labels_series.append(pose)
            img_batch.append(img_stacked_series)
            label_batch.append(labels_series)
        label_batch = np.array(label_batch)
        img_batch = np.array(img_batch)
        self.test_batch_pointer+=test_size
        return img_batch, label_batch        