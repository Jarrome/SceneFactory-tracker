import cv2
import torch
import numpy as np
from dpvo_api import get_dpvo
init_scale = 2

tracker = None

rgb_name = 'robotics_hall/color/%06d.png'
depth_name = 'robotics_hall/depth/%06d.png'
intrinsics = np.genfromtxt('robotics_hall/intrinsic.txt')
depth_im_scale = 1000.

intrinsics = torch.as_tensor(intrinsics).cuda(0)

def get_data(idx):
    rgb = cv2.imread(rgb_name%idx,-1)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_name%idx,-1)
    return rgb, depth

    
    


for idx in range(585, 1000, 1):
    print('frame',idx)
    # get data
    rgb, depth = get_data(idx)

    if tracker is None:
        H, W, _ = rgb.shape 
        tracker = get_dpvo(H-H%16, W-W%16, device='cuda:0', init_scale=init_scale)

    # process data
    rgb = torch.as_tensor(rgb[:H-H%16, :W-W%16, :]).permute(2,0,1).cuda(0)
    depth = torch.as_tensor(depth[:H-H%16, :W-W%16].astype(np.float32)/depth_im_scale).cuda(0) 

    # track

    with torch.no_grad():
        tracker(idx, rgb, intrinsics, depth)
    torch.cuda.empty_cache()


