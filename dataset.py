#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:02:08 2023

@author: choeyeong-a
"""
import skimage.color
import skimage.io
import skimage.transform

import numpy as np
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF


class SIPDataset(Dataset):
    def __init__(self, dataset_dir, subset, transforms=None):
        self.transforms = transforms
        self.subset = subset
        assert self.subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, self.subset)
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        self.annotations = list(annotations.values())
        
        self.image_idx=list(range(len(self.annotations)))
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_id = self.image_idx[idx]  # idx
        file_name = self.annotations[image_id]['filename']
        file_path = f'./dataset/{self.subset}/{file_name}'
        image = skimage.io.imread(file_path)
        height, width = image.shape[:2]
        
        annot = [a for a in self.annotations if a['filename']==file_name][0]
        
        # Add images
        if type(annot['regions']) is dict:
            shape_attributes = [r['shape_attributes'] for r in annot['regions'].values()]
        else:
            shape_attributes = [r['shape_attributes'] for r in annot['regions']]
        
        objects = [s['region_attributes'] for s in annot['regions'].values()]
        
        #print(objects)
        labels = []
        for n in objects:
            try:
                if n['name']=='fingerprint':
                    labels.append(1)
                elif n['name']=='identity_card':
                    labels.append(2)
                elif n['name']=='licenseplate':
                    labels.append(3)
                elif n['name']=='credit_card':
                    labels.append(4)
            except:
                pass
                
        info={'image_id':image_id,
              'file_name':file_name,
              'file_path':file_path,
              'height':height,
              'width':width,
              'shape_attributes':shape_attributes,
              'class_ids':labels
              }
        
        masks, labels = self.load_mask(info)
        boxes = self.load_boxe(info)
        
        area = boxes[:][2] * boxes[:][3]
        
        boxes[:][2] = boxes[:][0] + boxes[:][2]
        boxes[:][3] = boxes[:][1] + boxes[:][3]

        target = {'boxes': boxes, 'masks': masks, 'labels': labels}
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)  # TF.to_tensor
        
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).unsqueeze(0)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        
        return image, target

    def load_boxe(self, info):
        # [x,y,width,height]
        for i, p in enumerate(info["shape_attributes"]):
            if (p['name'] == "polygon"):
                # x, y: the upper-left coordinates of the bounding box
                # width, height: the dimensions of your bounding box
                x = np.min(p['all_points_x'])
                y = np.max(p['all_points_y'])
                h = y - np.min(p['all_points_y'])
                w = np.max(p['all_points_x']) - x
            elif (p['name'] == "ellipse"):
                h = p['ry']
                w = p['rx']
                y = p['cy'] + h/2
                x = p['cx'] - w/2
            
        return np.array([x,y,w,h], dtype=np.float32)

    def load_mask(self, info):
        class_ids = info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["shape_attributes"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["shape_attributes"]):
            if (p['name'] == "polygon"):
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif (p['name'] == "ellipse"):
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
                
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            
            mask[rr, cc, i] = 1

        class_ids = np.array(class_ids, dtype=np.int32) #Map class names to class IDs
        mask = np.array(mask, dtype=np.uint8)
        # print('load_mask',mask)
        
        return mask, class_ids

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)

        return image, target


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        # w, h = image.shape[:2]
        # image = image.resize(self.size)
        image = np.resize(image, self.size)

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        
        for i, v in enumerate(_masks):
            v = Image.fromarray(v).resize(self.size, resample=Image.BILINEAR)
            masks[i] = np.array(v, dtype=np.uint8)

        target['masks'] = masks
        
        return image, target
        

class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        
        return image, target