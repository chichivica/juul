# -*- coding: utf-8 -*-
"""
Сохраняем видео, которое считывается с камеры

Структура сохранения файлов

output dir:
    |-- day 1
        |-- video 1
        |-- video 2
        ...
    |-- day 2
    ...
"""

import cv2
from datetime import datetime as dt
import os, shutil, subprocess, sys
import time
import importlib
# custom
project_dir = os.path.realpath(os.path.dirname(__file__))
env = importlib.import_module('env', project_dir)


# get stage to set global variables
try:
    stage = sys.argv[1]
except IndexError:
    stage = 'test'
assert stage in env.ENVIRON.keys(), \
        '{} not amount available options {}'.format(stage, env.ENVIRON.keys())
# set global variables
VIDEO_OUT = env.ENVIRON[stage]['VIDEO_OUT']  # relative to project directory or absolute if starts with /
FINAL_DIR = env.ENVIRON[stage]['FINAL_DIR'] # it is copied here after finished writing
CAMERA = env.ENVIRON[stage]['CAMERA']
CODEC = env.ENVIRON[stage]['CODEC']
FORMAT = '.mp4'
FPS = env.ENVIRON[stage]['FPS']
SLEEP = 0.01
WINDOW_SIZE = 'infer' # set to infer or specify (width,height)
VIDEO_LEN = env.ENVIRON[stage]['VIDEO_LEN'] # in seconds, 3600 for hour-length videos (multiplied by fps later)
MAX_TIME = env.ENVIRON[stage]['MAX_TIME'] # what (hour,minute) to finish the script execution
REMOVE_IF_EXISTS = env.ENVIRON[stage]['REMOVE_IF_EXISTS'] # remove output folder if exists


def create_dir(dir_path, empty_if_exists=True):
    '''
    Creates a folder if not exists,
    If does exist, then empties it by default
    '''
    if os.path.exists(dir_path):
        if empty_if_exists:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


def open_stream_save(num_frames, write_path, camera_id, codec, fps, 
                     window_dims='infer'):
    '''
    Open camera and stream for num_frames
    Then save the stream
    '''
    cap = cv2.VideoCapture(camera_id)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    if window_dims == 'infer':
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        window_dims = (width, height)
    out = cv2.VideoWriter(write_path, fourcc, fps, window_dims)
    print('Starting to record', os.path.basename(write_path))
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    print('Video written to {}'.format(write_path))
    return write_path
    
    
def make_dir(base_dir, project_dir, remove_empty=True):
    '''
    Creates a dir for video collection
    '''
    folder_name = dt.today().strftime('%Y-%m-%d')
    if os.path.isabs(base_dir):
        write_dir = os.path.join(base_dir, folder_name)
    else:
        write_dir = os.path.join(project_dir, base_dir, folder_name)
    create_dir(write_dir, remove_empty)
    return write_dir
    
    
def get_path(write_dir, extension = '.mp4'):
    filename = str(int(time.time())) + extension
    write_path = os.path.join(write_dir, filename)
    return write_path


def calc_max_frames(end_hour, end_minute, fps):
    '''
    Calculate max number of frames before end time
    '''
    local = time.localtime()
    local_dt = dt(local.tm_year, local.tm_mon, local.tm_mday, 
                  local.tm_hour, local.tm_min, local.tm_sec)
    end_dt = dt(local.tm_year, local.tm_mon, local.tm_mday, 
                end_hour, end_minute, 0)
    return (end_dt - local_dt).total_seconds() * fps

if __name__ == '__main__':
    video_len = VIDEO_LEN*FPS
    # create output dirs
    write_dir = make_dir(VIDEO_OUT, project_dir, REMOVE_IF_EXISTS)
    final_dir = make_dir(FINAL_DIR, project_dir, REMOVE_IF_EXISTS)
    processes = []
    while True:
        # record and save
        write_path = get_path(write_dir, FORMAT)
        max_frames = calc_max_frames(MAX_TIME[0], MAX_TIME[1], FPS)
        num_frames = int(min(video_len, max_frames))
        filepath = open_stream_save(num_frames, write_path, CAMERA, CODEC, FPS)
        # move to final dir
        p = subprocess.Popen(['mv',filepath,final_dir], shell=False, 
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(p)
        time.sleep(SLEEP)
        # finish after due time
        if num_frames == max_frames:
            exit_codes = [p.wait() for p in processes]
            break