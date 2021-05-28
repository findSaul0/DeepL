import os
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import DeviceDataLoader
from DeviceDataLoader import DeviceDataLoader as dataLoader
from discriminator import Discriminator
from generator import Generator

# LEGGAIMO DAL FILE CSV TUTTI I NOMI DEGLI AUTORI
artist = pd.read_csv("/home/antoniofasulo/Documenti/archive/artists.csv")

# CI STAMPIAMO TUTTI I NOMI DEGLI AUTORI
def showNameArtisr():
    for i in artist['name']:
        print(i,"|")

batch_size = 128 # campioni da prendere per il training
image_size = (64,64) # size dell immagine
stats = (0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5)

# TRASFORMIAMO LE NOSTRE IMMAGINI
# STATS SERVE PER NORMALIZZARE I DATI
# ToTensor sono arry multidimensionali (nel nostro caso sono bidimensionali)
transform_ds = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize(*stats)])

# DEFINIAMO IL NOSTRO TRAIN SET
# IN QUESTO CASO TRASFORMIAMO LE IMMAGINI IN "MATRICI"
train_dataSet = torchvision.datasets.ImageFolder (root="/home/antoniofasulo/Documenti/archive/resized", transform= transform_ds)

# IL MODO IN CUI I DATI VERRANNO DATI AL DATALOADER
train_dataLoader = DataLoader(train_dataSet,batch_size,shuffle=True,num_workers=3,pin_memory=True)
print("Numero immagini:"+str(len(train_dataSet)))

#print(train_dataSet[1]) // vedi come diventano le immagini

# PER STAMPARE LE IMMAGINI ABBIAMO BISOGNO DI RISPETTARE LA CLASSE (vedi stampando train_dataSet
images,_ = train_dataSet[1912]
plt.imshow(images.permute(1,2,0))

# LE 3 FUNZIONI SOTTO SERVONO A FARE UNA GRIGLIA PER MOSTRARE LE IMMAGINI IN QUESTO CASO È UNA GRIGLIA (8x8)
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images,_ in dl:
        show_images(images, nmax)
        break

show_batch(train_dataLoader)
#plt.show()


device = DeviceDataLoader.get_default_device()
print(device)

train_dataLoader = dataLoader(train_dataLoader,device)

disciminatore = Discriminator()

#Tensore (e come se creassimo un immagine causuale)
x = torch.randn(batch_size,150,1,1)

generatore = Generator()

fake_images = generatore.forward(x)
print(fake_images.shape)
show_images(fake_images)
plt.show()

#PER OGGI BASTA
