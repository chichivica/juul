# -*- coding: utf-8 -*-
import importlib
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
from copy import deepcopy
import torch
import os


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch_utils = importlib.import_module('src.rgutils.torch', project_dir)
adience = importlib.import_module('src.age_gender.adience_utils', project_dir)


def load_vggface_model(arch, project_dir, weights_path):
    '''
    Load either a resnet50 or senet50 model
    with pre-trained VGGFace weights
    '''
    # load model
    if arch == 'resnet':
        resnet = importlib.import_module('src.VGGFace2-pytorch.models.resnet', project_dir)
        vggface = resnet.resnet50(num_classes=8631, include_top=True)
    elif arch == 'senet':
        senet = importlib.import_module('src.VGGFace2-pytorch.models.senet', project_dir)
        vggface = senet.senet50(num_classes=8631, include_top=True)
    else:
        raise Exception('Wrong architecture, should be resnet or senet.')
    # load weights into model
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
    own_state = vggface.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    return vggface


def customize_network(orig_net, num_classes):
    net = deepcopy(orig_net)
    in_feat = net.fc.in_features
    net.fc = nn.Linear(in_feat, num_classes)
    return net


def freeze_layers(net, trainable_children):
    for child,layers in net.named_children():
        if child in trainable_children:
            for param in layers.parameters():
                param.requires_grad = True
        else:
            for param in layers.parameters():
                param.requires_grad = False
    return net


def create_model(pretrained_network, num_classes, device, trainable_layers=None):
    net = customize_network(pretrained_network, num_classes)
    if trainable_layers:
        net = freeze_layers(net, trainable_layers)
    return net.to(device)


class AdienceDataset(Dataset):
    '''
    Load face image either with age or gender label
    '''
    def __init__(self, file_indices: list, label_type, labels_dir, data_dir, 
                 transforms=None, labels2int=True):
        assert label_type in ['gender', 'age'], 'Label type should be either gender or age'
        self.data = adience.load_labels(labels_dir, file_indices)
        accepted_label_values = {
                'age': ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', 
                        '(38, 43)', '(48, 53)', '(60, 100)'],
                'gender': ['f','m'],
                }
        self.data[label_type] = self.data[label_type].astype(str)
        self.data = self.data[self.data[label_type].isin(accepted_label_values[label_type])]
        if labels2int:
            self.classes_mapping = {k:v for v,k in enumerate(accepted_label_values[label_type])}
            self.data[label_type] = self.data[label_type].map(self.classes_mapping)
        self.label_type = label_type
        self.data_dir = data_dir
        self.transforms = transforms
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = adience.get_image_path(row, self.data_dir)[0]
        img = load_convert_resize(image_path, (256,256))
        label = row[self.label_type]
        if self.transforms:
            img = self.transforms(img)
        return img, label
                              
                              
def load_convert_resize(image_path, target_size):
    '''
    Load image with opencv, convert to rgb and resize
    '''
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

    