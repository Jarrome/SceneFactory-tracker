import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData

from .dpvo.utils import Timer
from .dpvo.dpvo import DPVO
from .dpvo.config import cfg
from .dpvo.stream import image_stream, video_stream
from .dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

pwd = os.path.dirname(os.path.realpath(__file__))

import argparse
args = argparse.Namespace()
args.network = pwd+'/dpvo.pth'
args.config = pwd+'/config/default.yaml'
cfg.merge_from_file(args.config)
cfg.BUFFER_SIZE = 2048

def get_dpvo(H,W,device='cuda:1'):
    global args, cfg 
    return DPVO(cfg, args.network, ht=H, wd=W, viz=False, device=device) 
    



