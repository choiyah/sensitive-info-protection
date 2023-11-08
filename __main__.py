#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:51:23 2023

@author: choeyeong-a
"""

from torch.utils.data import DataLoader

from model import MRCNN
from dataset import SIPDataset, Compose, Resize, ToTensor

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__=='__main__':
    classes = ('fingerprint', 'identity_card', 'licenseplate', 'credit_card')
    num_classes=4

    batch_size = 8 # 32
    lr = 1e-3
    max_size = 80
    num_workers = 2 # 8
    num_epochs = 1
    device = 'mps'
    hidden_layer = 8   # 256
    
    transforms_train = Compose([Resize((max_size, max_size)),ToTensor()])

    print('[*] Load dataset')
    dataset = SIPDataset('./dataset/', 'train', transforms=transforms_train)
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              collate_fn=collate_fn
                              )
    print('[*] Make model')
    model = MRCNN(num_classes, hidden_layer, classes, max_size, device)
    
    print('[*] Train dataset')
    model.train(num_epochs, train_loader, lr, device)
    
    
    
    