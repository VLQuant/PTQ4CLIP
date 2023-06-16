! pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

from imagenetv2_pytorch import ImageNetV2Dataset
import clip
from dataset import VLImageDataset
import torch
from tqdm import tqdm
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload,import_module
import multiprocessing
import os
import time
from itertools import product
sys.path.append("./PTQ4ViT")
import PTQ4ViT.utils.datasets as datasets
import PTQ4ViT.utils.net_wrap as net_wrap
from PTQ4ViT.utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from PTQ4ViT.utils.models import get_net
from dataset import VLImageDataset, VLLoader

model, preprocess = clip.load("ViT-B/32", device="cuda")


def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    print(pos/tot)
    return pos/tot


def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("PTQ4Vit/configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg


image_features, text_features, target = VLLoader(model, preprocess)

g = VLImageDataset(image_features, text_features, target)
calib_loader = torch.utils.data.DataLoader(g, batch_size=32, num_workers=1)



def experiment_basic(net='vit_base_patch16_384', config="PTQ4ViT"):
    """
    A basic testbench.
    """
    quant_cfg = init_config(config)
    net = model
    wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    # g=datasets.ViTImageNetLoaderGenerator('imagenetv2-matched-frequency-format-val','imagenet',32,32,16,kwargs={"model":net})
    # test_loader=g.test_loader()
    # calib_loader=g.calib_loader(num=32)


    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()
    
    # test_classification(net,test_loader)

experiment_basic()


