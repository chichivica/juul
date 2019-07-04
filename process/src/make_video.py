# -*- coding: utf-8 -*-
"""
From a processed video and saved data on bounding boxes, embeddings
generate a video with drawings:
    - detection zone
    - faces
    - *face landmarks
    - people ids
    - time between arrival and departure
    - separate colors for bypassers, actual visitors and employees
"""
import os, sys
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
#custom
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, draw_rectangles, load_pkl

INPUT_VIDEO = 'data/external/juul_stream/1556725808.mp4'
OUTPUT_FILE = 'data/interim/demo/demo_1556725808.mp4'

#DATA_FILE = 'data/interim/embeddings/demo_juul_2019-07-01.pkl'
DATA_FILE = 'data/interim/embeddings/corrected_demo_juul_2019-07-01.pkl'
#CLUSTER_LABELS = 'data/interim/clusters/demo_juul_2019-07-01.npy'
CLUSTER_LABELS = 'data/interim/clusters/corrected_demo_juul_2019-07-01.npy'
CLUSTER_TIMES = 'data/interim/clusters/demo_juul_2019-07-01_times.csv'
CLUSTER_DEMOGRAPHICS = 'data/interim/clusters/demographics_demo_juul_2019-07-01.pkl'
START_FRAME = 0
END_FRAME = None
START_X = 400
END_X = 2688-300
START_Y = 470
END_Y = 1520-250
CODEC = 'mp4v'


class DemoVideo:
    '''
    Load face boxes and embeddings and overlay them on video capture
    '''
    def __init__(self, data, input_file, output_file, cluster_labels,
                 cluster_times):
        for f in [data, input_file, cluster_labels,cluster_times]:
            assert os.path.exists(f), '{} does not exist'.format(f)
        out_dir = os.path.dirname(output_file)
        assert os.path.exists(out_dir), '{} dir not exists'.format(out_dir)
        with open(data, 'rb') as f:
            self.data = pickle.load(f)
        self.input_file = input_file
        self.output_file = output_file
        self.cluster_labels = np.load(cluster_labels)
        self.cluster_times = pd.read_csv(cluster_times)
        self.distinct_customers = 0
        self.reset_demographics()
        self.labels_so_far = set()
        self.label_times = {}
    
    def init_video_writer(self, codec, fps, window_dims, 
                          start_x, start_y, end_x, end_y):
        '''
        Create a video writer object
        '''
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.output_file, fourcc, fps,
                                      window_dims)
        self.fps = fps
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        
    def reset_demographics(self):
        self.gender_counts = {'male': 0, 'female': 0}
        self.age_counts = {'u-18': 0, '18-30': 0, '30-40': 0,
                           '40-50': 0, '50-65': 0, '65+': 0}

    @staticmethod
    def get_color(label):
        '''
        Return colors for different labels
        '''
        if label == -1: # bypassers in yellow
            return (0,255,255)
        elif label == -2: # empoyees in blue
            return (255,0,0)
        elif label == -3: # short timers in orange
            return (255,0,255)
        else: # visitors in green
            return (0,255,0)
        
    def update(self, labels_in_frame, frame_num):
        '''
        Update labels that have been seen so far to get unique
        '''
        # get unique customers
        pos_labs = labels_in_frame[labels_in_frame >= 0]
        self.labels_so_far.update(pos_labs)
        # set first and latest times
        for lab in pos_labs:
            lab_times = self.label_times.get(lab, {})
            if len(lab_times) == 0:
                lab_times['first'] = frame_num
                lab_times['last'] = frame_num
                lab_times['duration'] = 0
            else:
                lab_times['last'] = frame_num
                lab_times['duration'] = int((frame_num - lab_times['first']) / self.fps)
            self.label_times[lab] = lab_times    
        # get demographics count
        self.reset_demographics()
        for lab in self.labels_so_far:
            age = demographics[lab]['age']
            gender = demographics[lab]['gender']
            self.gender_counts[gender] += 1
            self.age_counts[age] += 1
        
    def draw_legend(self, frame):
        '''
        Put legend on images as follows:
            white for detection area
            green for customers
            blue for employees
            yellow/orange for bypassers
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), 
                              (255, 255, 255), thickness=2, lineType=4)
        # place unique counts
        text_x, text_y = self.end_x + 10, self.start_y + 25
        cv2.putText(frame, f'Visitors: {len(self.labels_so_far)}',
                    (text_x, text_y), font, 1.25, (0,255,0), 3)
        # place gender counts
        text_y += 90
        for gender,cnt in self.gender_counts.items():
            cv2.putText(frame, f'{gender:>6}: {cnt}',(text_x, text_y), font, 
                        1.25, (0,255,0), 3)
            text_y += 35
        # place age counts
        text_y += 70
        for age,cnt in self.age_counts.items():
            cv2.putText(frame, f'{age:>5}: {cnt}',(text_x, text_y), font, 
                        1.25, (0,255,0), 3)
            text_y += 35
        # place average length
        avg_len = np.mean([l['duration'] for l in self.label_times.values()])
        m,s = divmod(avg_len, 60)
        text_y += 70
        cv2.putText(frame, f'avg. visit len:',(text_x, text_y), font, 
                        1.25, (0,255,0), 3)
        text_y += 35
        cv2.putText(frame, f'{int(m):>2}:{int(s):>2}',(text_x, text_y), font, 
                        1.25, (0,255,0), 3)
        # place colored rectangle legend
        legend = [('Detection area',(255, 255, 255)),('Visitor',(0, 255, 0)),
                  ('Employee',(255, 0, 0)),('Bypasser',(0, 255, 255))]
        text_x, text_y = self.start_x, self.end_y + 100
        for text,color in legend:
            cv2.putText(frame, text, (text_x, text_y), font, 1, color, 1)
            (text_w,text_h),_ = cv2.getTextSize(text, font, 1, 1)
            p1 = (text_x - 10, text_y - text_h - 10)
            p2 = (text_x + text_w + 10, text_y + 10)
            cv2.rectangle(frame, p1, p2, color, thickness=2, lineType=4)
            text_x = text_x + text_w + 30
            
    def pretty_minutes_seconds(self, labels):
        '''
        Return for given labels a list with their values as MM:SS
        Empty for labels are less than 0
        '''
        pretty_lens = []
        for lab in labels:
            lab_times = self.label_times.get(lab, {})
            if len(lab_times) == 0:
                pretty_lens.append('')
            else:
                m,s = divmod(lab_times['duration'], 60)
                pretty_lens.append(f'{m:>2}:{s:>2}')
        return pretty_lens
        
    def create_video(self, start_frame, end_frame):
        '''
        Create an output video with detections
        '''
        cap = cv2.VideoCapture(self.input_file)
        # get data
        face_frames_inds = np.array(self.data['indices'])[:,0]
        rects = np.array(self.data['boxes'])
        confidences = np.array(self.data['scores'])
        # adjust boxes
        rects[:,0::2] = rects[:,0::2] + self.start_x
        rects[:,1::2] = rects[:,1::2] + self.start_y
        # iterate over frames
        num_frames = face_frames_inds.max() if end_frame is None else end_frame
        for i, frame in tqdm(enumerate(range(num_frames))):
            ret,frame = cap.read()
            if not ret:
                print('Could not read frame', i)
                break
            # skip if has not reached startyet, break if beyond the end
            if i < start_frame: continue
            # place rectangulars on this frame
            matches = np.argwhere(face_frames_inds == i).squeeze(1)
            if len(matches) > 0:
                rect = rects[matches]
#                conf = confidences[matches]
                cluster_labels = self.cluster_labels[matches]
                self.update(cluster_labels, i)
                colors = [self.get_color(l) for l in cluster_labels]
                pretty_durations = self.pretty_minutes_seconds(cluster_labels)
                draw_rectangles(frame, rect, colors, texts=pretty_durations, )
            self.draw_legend(frame)
            self.writer.write(frame)
        self.writer.release()
        cap.release()
        
if __name__ == '__main__':
    # prepare paths
    data_file = get_abs_path(__file__, DATA_FILE, depth=2)
    input_video = get_abs_path(__file__, INPUT_VIDEO, depth=2)
    output_file = get_abs_path(__file__, OUTPUT_FILE, depth=2)
    cluster_labels = get_abs_path(__file__, CLUSTER_LABELS, depth=2)
    cluster_times = get_abs_path(__file__, CLUSTER_TIMES, depth=2)
    create_dir(os.path.dirname(output_file), False)
    cluster_demographics = get_abs_path(__file__, CLUSTER_DEMOGRAPHICS, depth=2)
    # create video
    demographics = load_pkl(cluster_demographics)
    demo = DemoVideo(data_file, input_video, output_file, cluster_labels,
                     cluster_times)
    demo.init_video_writer(CODEC, 20.0, (2688,1520), START_X, START_Y, END_X, END_Y)
    demo.create_video(START_FRAME, END_FRAME, )