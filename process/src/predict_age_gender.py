# -*- coding: utf-8 -*-
"""
Predict age and gender for each cluster
dased on all images assigned to that cluster.
Save results in a file for db injection
"""

import os,sys
import importlib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# custom modules
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if project_dir not in sys.path: sys.path.insert(0, project_dir)
agender_infer = importlib.import_module('src.age_gender.inference_utils', project_dir)
agender_train = importlib.import_module('src.age_gender.train_utils', project_dir)
from src.env import configs
from src.utils import get_cmd_argv, get_abs_path, create_dir

DEPTH = 2

if __name__ == '__main__':
    # get configs and paths
    q_date = get_cmd_argv(sys.argv, 1, None)
    q_name = get_cmd_argv(sys.argv, 2, 'test')
    detections_path = get_abs_path(__file__, configs['WRITE_DETECTIONS'].format(name=q_name,
                                               detector=configs['DETECTOR']), DEPTH)
    clusters_path = get_abs_path(__file__, configs['WRITE_CLUSTERS'].format(name=q_name), DEPTH)
    write_path = get_abs_path(__file__, configs['WRITE_DEMOGRAPHICS'].format(name=q_name, 
                                                          date=q_date), DEPTH)
    create_dir(os.path.dirname(write_path), False)
    vggface_weights = get_abs_path(__file__, configs['DEMOGRAPHIC']['vggface'], DEPTH)
    age_weights = get_abs_path(__file__, configs['DEMOGRAPHIC']['age'], DEPTH)
    gender_weights = get_abs_path(__file__, configs['DEMOGRAPHIC']['gender'], DEPTH)
    # on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug = configs['REQUIRE_PRINTS']
    # load models
    cv_indices = list(range(configs['DEMOGRAPHIC']['num_folds']))
    vggface = agender_train.load_vggface_model(configs['DEMOGRAPHIC']['vggface_backbone'],
                                           project_dir, vggface_weights).to(device)
    age_cv = agender_infer.load_cv_models(vggface, age_weights, cv_indices, 
                                          configs['DEMOGRAPHIC']['age_out_classes'],
                                          device, print_results=debug)
    gender_cv = agender_infer.load_cv_models(vggface, gender_weights, cv_indices, 
                                          configs['DEMOGRAPHIC']['gender_out_classes'],
                                          device, print_results=debug)
    # create face loader
    img_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.45, 0.44, 0.49], [0.25, 0.22, 0.24]),
                ])
    face_dataset = agender_infer.FaceDataset(detections_path, clusters_path, 
                                           configs['DEMOGRAPHIC']['input_size'],
                                           input_transforms=img_transforms,
                                           print_results=debug)
    face_loader = DataLoader(face_dataset, shuffle=False, num_workers=4,
                             batch_size=configs['DEMOGRAPHIC']['batch_size'])
    # predict and decode
    age_predictions, gender_predictions = \
            agender_infer.batch_cv_predictions(age_cv, gender_cv, face_loader,
                                               device)
    predicted_age = np.array(list(map(agender_infer.get_age_fn, 
                                       age_predictions)))
    predicted_gender = np.array(list(map(agender_infer.get_gender_fn, 
                                         gender_predictions)))
    # aggregate and transform predictions for each class
    predictions = pd.DataFrame({
                        'cluster': face_loader.dataset.clusters,
                        'age_counter': predicted_age,
                        'gender_counter': predicted_gender
                        })
    cluster_demographics = predictions.groupby('cluster', as_index=False)\
                    .agg({
                'age_counter': lambda x: agender_infer.cluster_label_counter(x),
                'gender_counter': lambda x: agender_infer.cluster_label_counter(x),
                    })
    cluster_demographics['age'] = \
        cluster_demographics['age_counter'].apply(lambda x: agender_infer.transform_age(x))
    cluster_demographics['gender'] = \
        cluster_demographics['gender_counter'].apply(lambda x: agender_infer.transform_gender(x))
    cluster_demographics.to_csv(write_path, index=False)
    print(f'Cluster demographics saved to {write_path}, '
          f'num rows {len(cluster_demographics)}')
    