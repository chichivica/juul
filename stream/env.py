# -*- coding: utf-8 -*-
"""
Environment variables
"""

ENVIRON = {
        'docker': {
                'FPS': 20,
                'CAMERA': 'rtsp://admin:41cdf8b4@192.168.0.2/Streaming/Channels/101',
                'CODEC': 'X264',
                'VIDEO_LEN': 1800,
                'REMOVE_IF_EXISTS': False,
                'VIDEO_OUT': 'tmp/',
                'FINAL_DIR': '/data/juul/',
                'MAX_TIME': (23,1),
                },
        'test': {
                'FPS': 20,
                'CAMERA': 'rtsp://10.1.1.183:554/1/h264major',
                'CODEC': 'mp4v',
                'VIDEO_LEN': 15,
                'REMOVE_IF_EXISTS': True,
                'VIDEO_OUT': 'tmp/',
                'FINAL_DIR': 'data/',
                'MAX_TIME': (14,59),
                }
        }