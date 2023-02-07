from torch import nn
import torch
import os
import urllib
from torchvision import transforms

def get_model_path(model_name):
    model_file = model_name + '.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = 'https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/' + model_file + '?raw=true'
        print('Downloading', model_name, 'from', url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


def load_model():
    model = torch.load('app/first.torch', map_location=torch.device('cpu'))
    return model.to(get_device())

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
  ])