# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import os,sys
import importlib
# custom modules
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if project_dir not in sys.path: sys.path.insert(0, project_dir)
train_utils = importlib.import_module('src.age_gender.train_utils', project_dir)
utils = importlib.import_module('src.utils', project_dir)


age_vals =  ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', 
            '(38, 43)', '(48, 53)', '(60, 100)']
gender_vals = ['f', 'm']
base_age_mapping = {'(0, 2)':'0-17', '(4, 6)':'0-17', '(8, 13)':'0-17',
                    '(15, 20)':'18-25', '(25, 32)':'25-35', '(38, 43)':'35-45',
                    '(48, 53)':'45-60', '(60, 100)':'60+'}
base_gender_mapping = {'f': 'female', 'm': 'male'}


def load_cv_models(backbone_model, model_path, indices, num_classes, device, 
                   print_results=False):
    '''
    Load cross-validated models from disk and create models for inference.
    Params:
        backbone_model - loaded vgg model
        model_path - path to a cv model with '{index}' inside
        indices - list of cv indices
        num_classes - 8 for age and 1 for gender
    Returns a dict in the format {cv_index: model}
    '''
    cv_models = {}
    for i in indices:
        path = model_path.format(index = i)
        net = train_utils.create_model(backbone_model, num_classes, device,)
        net.load_state_dict(torch.load(path))
        net.eval()
        cv_models[i] = net
        if print_results:
            print(f'{device.type}:{device.index}: Loaded {path}')
    return cv_models


class FaceDataset(Dataset):
    '''
    Use this dataset class at inference
    Params:
        face_detections: path to hdf5 file with detections from face_detector.py
        face_clusters: path to npy file with assigned cluster from cluster_faces.py
        model_input_size: resize images for model input, used (224,224)
        input_transforms: torch transforms as traing
    '''
    def __init__(self, face_detections, face_clusters, model_input_size,
                 input_transforms=None, print_results=False):
        image_paths = utils.load_hdf(face_detections, keys=['image_paths'], 
                                     print_results=print_results)['image_paths']
        clusters = np.load(face_clusters)
        assert len(image_paths) == len(clusters)
        # select only clustered images
        self.image_paths = image_paths[clusters >= 0]
        self.image_paths = [p.decode('utf-8') for p in self.image_paths]
        self.clusters = clusters[clusters >= 0]
        assert len(self.image_paths) == len(self.clusters)
        self.model_input_size = model_input_size
        self.input_transforms = input_transforms
        if print_results:
            print(f'{self.__len__()} face images in the dataset')
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = train_utils.load_convert_resize(image_path, self.model_input_size)
        if self.input_transforms:
            img = self.input_transforms(img)
        return img


def batch_cv_predictions(age_cv, gender_cv, face_loader, device, 
                         gender_threshold=0.5):
    '''
    Make batch predictions for faces both age and gender
    Params:
        age_cv: dict of cv age models
        gender_cv: dict of cv gender models
        face_loader: dataloader for faces
        gender_threshold: 0.5 default where 1 - male, 0 - female
    Returns two numpy arrays for age and gender predictions,
    each one of size (len(dataset), len(cv_models))
    '''
    dataset_len = len(face_loader.dataset)
    age_predictions = torch.zeros(dataset_len, len(age_cv), device=device)
    gender_predictions = torch.zeros(dataset_len, len(gender_cv), device=device)
    batch_num = 0
    with torch.no_grad():
        for x in tqdm(face_loader, desc='Batches'):
            start = batch_num * face_loader.batch_size
            end = min((batch_num + 1) * face_loader.batch_size, dataset_len)
            # predict ages
            for i,age_model in age_cv.items():
                age_out = age_model(x.to(device))
                _,y_pred_age = torch.max(age_out, dim=1)
                age_predictions[start:end, i] = y_pred_age
            # predict genders
            for i,gender_model in gender_cv.items():
                gender_out = gender_model(x.to(device))
                y_pred_gender = torch.sigmoid(torch.squeeze(gender_out, dim=1)) > gender_threshold
                gender_predictions[start:end, i] = y_pred_gender
            batch_num += 1
    return age_predictions, gender_predictions


# count predictions for age and gender
age_decoded = {v:k for v,k in enumerate(age_vals)}
gender_decoded = {v:k for v,k in enumerate(gender_vals)}
get_age_fn = lambda x: Counter([age_decoded[int_label] for int_label in list(x.cpu().numpy())])
get_gender_fn = lambda x: Counter([gender_decoded[int_label] for int_label in list(x.cpu().numpy())])


def transform_age(age_counter):
    '''
    Transform model age output to desired groups:
    0-17 18-25 25-35 35-45 45-60 60+
    Return a str age group
    '''
    if len(age_counter) == 1:
        key = list(age_counter.keys())[0]
        return base_age_mapping[key]
    elif len(age_counter) > 1:
        (top1_label,top1_cnt),(top2_label,top2_cnt) = age_counter.most_common(2)
        if top1_label != '(15, 20)':
            cnter = Counter({top1_label: top1_cnt})
        else:
            if top2_label in ['(0, 2)', '(4, 6)', '(8, 13)',]:
                cnter = Counter({top2_label: top1_cnt + top2_cnt})
            elif top2_label in ['(15, 20)', '(25, 32)', 
                                '(38, 43)', '(48, 53)', '(60, 100)']:
                cnter = Counter({top1_label: top1_cnt + top2_cnt})
            else:
                raise Exception(f'Invalid label: {top2_label} in {age_counter}')
        return transform_age(cnter)
    else:
        raise Exception(f'{age_counter} has len {len(age_counter)}')
        
        
def transform_gender(gender_counter):
    '''
    Return most frequent label from a counter
    '''
    lab,cnt = gender_counter.most_common(1)[0]
    return base_gender_mapping[lab]
        
        
def cluster_label_counter(labels):
    '''
    From an array of labels return a counter of labels
    '''
    label_counter = Counter()
    _ = [label_counter.update(mc) for mc in labels]
    return label_counter

