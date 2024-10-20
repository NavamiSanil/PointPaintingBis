import os
import time
import enum
import argparse
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
from utils.label import labels, id2label
from utils.utils import TransformationTrain

class KittiSemanticDataset(Dataset):
    def __init__(self, root='data/KITTI', split='train', mode='semantic', transform=None, transform_train=None):
        self.transform = transform
        self.transform_train = transform_train

        assert split in ['train', 'test']
        self.split = 'training' if split == 'train' else 'testing'

        self.root = os.path.join(root, self.split)

        assert mode in ['semantic', 'color']
        self.mode = mode

        # paths of images and labels 
        self.imagesPath = os.path.join(self.root, "image_2")
        self.semanticPath = os.path.join(self.root, "semantic")
        self.colorPath = os.path.join(self.root, "semantic_rgb")

        # list all images / labels paths
        images_names = sorted(os.listdir(self.imagesPath))
        semantic_names = sorted(os.listdir(self.semanticPath))
        color_names = sorted(os.listdir(self.colorPath))

        # add the root path to images names
        self.images_paths = [os.path.join(self.imagesPath, name) for name in images_names]
        self.semantic_paths = [os.path.join(self.semanticPath, name) for name in semantic_names]
        self.color_paths = [os.path.join(self.colorPath, name) for name in color_names]

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        semantic_path = self.semantic_paths[index]
        color_path = self.color_paths[index]

        image = self.read_image(image_path)
        semantic = None

        if self.mode == 'semantic':
            semantic = self.read_image(semantic_path)
        elif self.mode == 'color':
            semantic = self.read_image(color_path)

        image = np.asarray(image)
        semantic = np.asarray(semantic)

        # its 3 identical channels (each one is semantic map)
        if self.mode == 'semantic':
            semantic = semantic[:, :, 0]

        shape = (1024, 512)
        image = cv2.resize(image, shape)
        semantic = cv2.resize(semantic, shape, interpolation=cv2.INTER_NEAREST)

        if self.split == 'training':
            semantic = self.remove_ignore_index_labels(semantic)

        if self.transform_train:
            image_label = self.transform_train(dict(im=image, lb=semantic))
            image = image_label['im'].copy()
            semantic = image_label['lb'].copy()

        if self.transform:
            image = self.transform(image)

        return image, semantic

    def __len__(self):
        return len(self.images_paths)

    def read_image(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def remove_ignore_index_labels(self, semantic):
        for id in id2label:
            label = id2label[id]
            trainId = label.trainId
            semantic[semantic == id] = trainId
        return semantic

# Continue with the rest of your dataset functions...
