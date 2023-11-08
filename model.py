#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:02:18 2023

@author: choeyeong-a
"""
import tqdm

import torch
# import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            
class MRCNN():
    def __init__(self, num_classes, hidden_layer, classes, max_size, device):
        self.num_classes = num_classes
        self.hidden_layer = hidden_layer
        self.classes = classes
        self.max_size = max_size
        self.device = device
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)
        
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
        
        self.in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask, self.hidden_layer, len(self.classes)+1)
        
        self.model.to(device)
        
    def train(self, num_epochs, train_loader, lr, device):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
        
        self.model.train()
            
        for epoch in range(num_epochs):
            # Setting the tqdm progress bar
            data_iter = tqdm.tqdm(enumerate(train_loader),
                                  desc="EPOCH:%d" % (epoch),
                                  total=len(train_loader),
                                  bar_format="{l_bar}{bar:10}{r_bar}")
            
            for i, (images, targets) in data_iter:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)  # 모델 손실 계산
                loss = sum(loss for loss in loss_dict.values())
                
                print(
                    f"[{i}/{epoch}] Clf loss: {loss_dict['loss_classifier'].item():.5f},"\
                        f" Mask loss: {loss_dict['loss_mask'].item():.5f}, "\
                            f"Box loss: {loss_dict['loss_box_reg'].item():.5f}, "\
                                f"Obj loss: {loss_dict['loss_objectness'].item():.5f}, "\
                                    f"Total loss: {loss.item():.5f}")
                loss.backward()
                optimizer.step()
                
                torch.save(self.model.state_dict(), f"model_ep{i}.pt")
                print(f"Save model: model_ep{i}.pt")
    