# -*- coding: utf-8 -*-
"""
Get image paths for cropped faces,
derive video name and frame number and 
retrive the whole frame from video and 
save for visual analysis
"""

import cv2
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import psycopg2
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path: sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, get_cmd_argv, draw_rectangles
from src.env import configs


FILE_DEPTH = 2


def get_video_path(full_path, video_dir, extension='mp4'):
    '''
    From a full path of a cropped image get video path
    '''
    video_name = os.path.basename(full_path).split('_')[0]
    video_name = '.'.join([video_name, extension])
    video_path = os.path.join(video_dir, video_name)
    assert os.path.exists(video_path), f'{video_path} does not exist'
    return video_path


def get_frame_number(full_path):
    '''
    From a full path of a cropped image get frame number
    '''
    frame_num = os.path.basename(full_path).split('_')[1]
    return int(frame_num)


def grab_retrieve_frames(video_path, frames, out_dir, video_boxes,
                         x_offset=0, y_offset=0,):
    '''
    Skip thru a video and save specified frames as images
    drawing bounding boxes on the fly
    '''
    frames = sorted(frames)
    cap = cv2.VideoCapture(video_path)
    for f in tqdm(frames, desc='Frames'):
        ret = cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        if not ret:
            print(f'Could not read {f} frame from {video_path}')
            break
        else:
            ret,frame = cap.read()
            if ret:
                filepath = make_new_path(video_path, f, out_dir)
                image_boxes = video_boxes.loc[(video_boxes['frame'] == f), 
                                              'box']
                image_boxes = np.array(list(map(eval, image_boxes)))
                image_boxes[:,::2] += x_offset
                image_boxes[:,1::2] += y_offset
                draw_rectangles(frame, image_boxes)
                cv2.imwrite(filepath, frame)
            else:
                print(f'Could not read {f} frame from {video_path}')
                break
    cap.release()
    
    
def make_new_path(video_path, frame_number, out_dir, image_extension='jpg'):
    '''
    Create a new destination path for saving images
    '''
    video_name = os.path.basename(video_path).split('.')[0]
    filename = '_'.join([str(video_name), str(frame_number)])
    filename = '.'.join([filename, image_extension])
    filepath = os.path.join(out_dir, filename)
    return filepath
    

if __name__ == '__main__':
    # get configs
    q_name = get_cmd_argv(sys.argv, 2, 'test')
    q_date = get_cmd_argv(sys.argv, 1, None)
    assert q_date is not None, 'provide date as 2nd argument'
    video_dir = get_abs_path(__file__, configs['VIDEO_PATH'].format(date=q_date),
                                 depth=FILE_DEPTH)
    data_path = get_abs_path(__file__, configs['WRITE_RESULTS'].format(name=q_name,
                                                                       date=q_date),
                             depth=FILE_DEPTH)
    out_dir = get_abs_path(__file__, configs['WRITE_FRAMES'].format(name=q_name,
                                                                    date=q_date), 
                            depth=FILE_DEPTH)
    create_dir(out_dir, True)
    top_adjust = configs['CROP_FRAMES']['top']
    left_adjust = configs['CROP_FRAMES']['left']
    # load data and get video paths + frame numbers
#    conn = psycopg2.connect(**configs['DB_CONNECTION'])
#    query = '''
#            SELECT * FROM visitors
#            WHERE pk_id > 280
#            '''
#    data = pd.read_sql(query, conn)
    data = pd.read_csv(data_path)
    data['video_begin'] = data['first_image'].apply(lambda x: get_video_path(x, video_dir))
    data['video_end'] = data['last_image'].apply(lambda x: get_video_path(x, video_dir))
    data['frame_begin'] = data['first_image'].apply(lambda x: get_frame_number(x))
    data['frame_end'] = data['last_image'].apply(lambda x: get_frame_number(x))
#    data['video_begin'] = data['photo_begin'].apply(lambda x: get_video_path(x, video_dir))
#    data['video_end'] = data['photo_end'].apply(lambda x: get_video_path(x, video_dir))
#    data['frame_begin'] = data['photo_begin'].apply(lambda x: get_frame_number(x))
#    data['frame_end'] = data['photo_end'].apply(lambda x: get_frame_number(x))    
    # combine and get unique videos with frames
    videos = pd.DataFrame({
            'path': data['video_begin'].append(data['video_end']),
            'frame': data['frame_begin'].append(data['frame_end']),
            'box': data['first_box'].append(data['last_box']),
            })
    video_frames = videos.groupby('path',).agg({'frame': 'unique'})
    for path,row in tqdm(video_frames.iterrows(), desc='Videos'):
        video_boxes = videos.loc[videos['path'] == path, ['frame','box']]
        grab_retrieve_frames(path, row['frame'], out_dir, video_boxes,
                             x_offset=left_adjust, y_offset=top_adjust)
    # write new file
    data['photo_begin'] = data.apply(lambda x: make_new_path(x['video_begin'], 
                                        x['frame_begin'], out_dir),
                                     axis=1)
    data['photo_end'] = data.apply(lambda x: make_new_path(x['video_end'], 
                                        x['frame_end'], out_dir),
                                     axis=1)
    bex = data['photo_begin'].apply(lambda x: os.path.exists(x))
    eex = data['photo_end'].apply(lambda x: os.path.exists(x))
    print(f'{sum(bex) + sum(eex)} out of {len(bex) + len(eex)} photos exist')
    data[['cluster','time_min','time_max','photo_begin','photo_end']]\
            .to_csv(data_path, index=False)