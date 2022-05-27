
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
from scipy.io import loadmat
import os
from sklearn.model_selection import train_test_split
import cv2

"""**Data Preparetion**"""

read_for = open("/content/dataset/annotations/annotations/list.txt")
sample_files=[]
class_decode = {}
sample_labels=[]
for txt_line in read_for:
    if txt_line[0] == '#':
        continue
    file_name, clsaa_id, cat_dog_id, sub_class_id = txt_line.split(' ')
    if cat_dog_id == '1':
        sample_files.append(os.path.join("./dataset/images/",file_name + '.jpg'))
        sample_labels.append(sub_class_id)
        class_decode[int(sub_class_id.strip("\n"))] = file_name.split("_")[0]
read_for.close()
sample_labels = [int(elem.strip("\n")) for elem in sample_labels]

class_decode

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class MyImageDataset(Dataset):
    
  def __init__(self, labels, list_IDs, transforms):
    self.labels = labels
    self.list_IDs = list_IDs
    self.transforms=transforms
    
  def __len__(self):
    return len(self.labels)

  def __getitem__(self,i):
    img = Image.open(self.labels[i])
    img = img.convert('RGB')
    img = self.transforms(img)
    y = self.list_IDs[i]
    return img, y

def show_batch(dataloader, rows, columns):
    data=iter(dataloader)
    fig = plt.figure(figsize=(15, 12))
    
    imgs, _=data.next()
    

    for i in range(rows*columns):
        npimg=imgs[i].numpy()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.axis('off')
    plt.show()

transform=transforms.Compose([
                              transforms.Resize((299,299)),
                              transforms.ToTensor(),
                              # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])

sample_ds = MyImageDataset(sample_files, sample_labels, transform)
sample_dl = DataLoader(sample_ds,batch_size=8) # , shuffle=True

show_batch(sample_dl, 2, 4)

x_train, x_test, y_train, y_test = train_test_split(sample_files, sample_labels, random_state=42)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = "cpu"
# torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 2}

train_set = MyImageDataset(x_train, y_train, transform)
train_gen = iter(torch.utils.data.DataLoader(train_set, **params))

val_set = MyImageDataset(x_test, y_test, transform)
val_gen = iter(torch.utils.data.DataLoader(val_set, **params))

"""**Model learning**"""

!pip install timm

import timm
from tqdm import tqdm
model = torch.nn.Sequential(*(list(timm.create_model('efficientnet_b7',
                                                     pretrained=True,
                                                     num_classes=1000).
                                   children())[:-1]))
model.eval()

def extract_features(model, max_epochs=1):
  train_f = []
  train_lbl = []
  val_f = []
  val_lbl = []
  with torch.no_grad():
    for epoch in range(max_epochs):
    #   # Training
      for _ in tqdm(range(len(train_gen))):
        local_batch, local_labels = train_gen.next()
    #     # Transfer to GPU
    #     # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        # print(model.extract_features(local_batch))
        train_f.append(model(local_batch))
        train_lbl.append(local_labels)

        # Validation
      for local_batch, local_labels in tqdm(val_gen):
          # Transfer to GPU
          # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

          # Model computations
          val_f.append(model(local_batch))
          val_lbl.append(local_labels)
    
  # train_f = np.vstack(train_f)
  # train_lbl = np.vstack(train_lbl)
  # val_f = np.vstack(val_f)
  # val_lbl = np.vstack(val_lbl)
  return [train_f, train_lbl], [val_f, val_lbl]

tr, v = extract_features(model, max_epochs=25)

train_f = np.vstack([elem.numpy() for elem in tr[0][:-1]])
train_l = np.hstack([elem.numpy() for elem in tr[1][:-1]])
val_f = np.vstack([elem.numpy() for elem in v[0][:-1]])
val_l = np.hstack([elem.numpy() for elem in v[1][:-1]])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
clf.fit(train_f, train_l)

clf.score(val_f, val_l)

from sklearn.metrics import roc_auc_score

roc_auc_score(val_l, clf.predict_proba(val_f), multi_class="ovo")

!pip install catboost
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=20)

model.fit(train_f, train_l)

roc_auc_score(val_l, model.predict_proba(val_f), multi_class="ovo")

