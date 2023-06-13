import numpy as np
import torch
import clip
from tqdm import tqdm
import json
import utils
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang_bits", type=int, help="language bitwidth")
args = parser.parse_args()

#print("Torch version:", torch.__version__)
#print("Clip Models:", clip.available_models())

model, preprocess = clip.load("RN50")


imagenet_classes = json.load(open('./class_imagenet.json','r'))['imagenet_classes']

#print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

from imagenetv2_pytorch import ImageNetV2Dataset
images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=1)
print("start")
with torch.no_grad():
  v32=[]
  for i, (images, target) in enumerate(tqdm(loader)):
      image_input = images
      targets = target.tolist()
      texts = ["a photo of a {}".format(imagenet_classes[i]) for i in targets]
      text_input = clip.tokenize(texts)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      image_input = image_input.to(device)
      text_input = text_input.to(device)
      model = model.to(device)
      text_features = model.encode_text(text_input)
      image_features = model.encode_image(image_input)

      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      text_features = text_features / text_features.norm(dim=-1, keepdim=True)

      clip_score = torch.matmul(image_features, text_features.T)
      result_list = clip_score.diagonal().tolist()
      result_list = list(np.round(np.array(result_list)*100,2))
      v32 += result_list
  print(len(v32))

#pd.Series(cos_list).to_csv("saver.csv",index=False, header=False)

with utils.qutils.QuantizationEnabler(model, 'symmetric per_layer', 'symmetric per_layer', 8, silent=True):
 with torch.no_grad():
  v8=[]
  for i, (images, target) in enumerate(tqdm(loader)):
      image_input = images
      targets = target.tolist()
      texts = ["a photo of a {}".format(imagenet_classes[i]) for i in targets]
      text_input = clip.tokenize(texts)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      image_input = image_input.to(device)
      text_input = text_input.to(device)
      model = model.to(device)
      text_features = model.encode_text(text_input)
      image_features = model.encode_image(image_input)

      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      text_features = text_features / text_features.norm(dim=-1, keepdim=True)

      clip_score = torch.matmul(image_features, text_features.T)
      result_list = clip_score.diagonal().tolist()
      result_list = list(np.round(np.array(result_list)*100,2))
      v8 += result_list
  print(len(v8))

with utils.qutils.QuantizationEnabler(model, 'symmetric per_layer', 'symmetric per_layer', 4, silent=True):
 with torch.no_grad():
  v4=[]
  for i, (images, target) in enumerate(tqdm(loader)):
      image_input = images
      targets = target.tolist()
      texts = ["a photo of a {}".format(imagenet_classes[i]) for i in targets]
      text_input = clip.tokenize(texts)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      image_input = image_input.to(device)
      text_input = text_input.to(device)
      model = model.to(device)
      text_features = model.encode_text(text_input)
      image_features = model.encode_image(image_input)

      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      text_features = text_features / text_features.norm(dim=-1, keepdim=True)

      clip_score = torch.matmul(image_features, text_features.T)
      result_list = clip_score.diagonal().tolist()
      result_list = list(np.round(np.array(result_list)*100,2))
      v4 += result_list
  print(len(v4))

with utils.qutils.QuantizationEnabler(model, 'symmetric per_layer', 'symmetric per_layer', 2, silent=True):
 with torch.no_grad():
  v2=[]
  for i, (images, target) in enumerate(tqdm(loader)):
      image_input = images
      targets = target.tolist()
      texts = ["a photo of a {}".format(imagenet_classes[i]) for i in targets]
      text_input = clip.tokenize(texts)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      image_input = image_input.to(device)
      text_input = text_input.to(device)
      model = model.to(device)
      text_features = model.encode_text(text_input)
      image_features = model.encode_image(image_input)

      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      text_features = text_features / text_features.norm(dim=-1, keepdim=True)

      clip_score = torch.matmul(image_features, text_features.T)
      result_list = clip_score.diagonal().tolist()
      result_list = list(np.round(np.array(result_list)*100,2))
      v2 += result_list
  print(len(v2))

col_names = ['v32l'+str(args.lang_bits), 'v8l'+str(args.lang_bits), 'v4l'+str(args.lang_bits), 'v2l'+str(args.lang_bits)]
sheetname = "lang"+str(args.lang_bits)
pd.DataFrame({col_names[0]:v32, col_names[1]:v8, col_names[2]:v4, col_names[3]:v2}).to_excel("imagenet_cos.xlsx",index=False, header=True, sheet_name=sheetname)










































