
import os
import json
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import clip
import copy
import torch
import numpy as np
from imagenetv2_pytorch import ImageNetV2Dataset

def VLLoader(model, preprocess, num = 32):
    imagenet_classes = json.load(open('./imagenet_classes.json','r'))['imagenet_classes']
    images = ImageNetV2Dataset(transform=preprocess)
    # loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=1)
    inds=np.random.permutation(len(images))[:num]
    print(inds)
    calib_set=torch.utils.data.Subset(copy.deepcopy(images),inds)
    loader = torch.utils.data.DataLoader(calib_set, batch_size=32, num_workers=1)

    # for i, (images, target) in enumerate(tqdm(loader)):
    images, target = next(iter(loader))
    image_input = images

    targets = target.tolist()
    texts = ["a photo of a {}".format(imagenet_classes[i]) for i in targets]
    text_input = clip.tokenize(texts)

    text_features = model.encode_text(text_input.cuda())
    image_features = model.encode_image(image_input.cuda())

    return image_features, text_features, target


class VLImageDataset(Dataset):
    def __init__(self, image_features, text_features, target):
        self.image_features = image_features
        self.text_features = text_features
        self.target = target

    def __len__(self):
        return self.text_features.shape[0]

    def __getitem__(self, idx):
        image = self.image_features[idx]
        text = torch.tensor([self.text_features[idx] for x in self.__len__()])
        target = self.target[idx]
        return image, text, target

# model, preprocess = clip.load("ViT-B/32", device = "cuda")
# VLLoader(model, num = 32)
