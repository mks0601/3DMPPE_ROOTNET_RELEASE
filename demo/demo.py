import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_pose_net(cfg, False)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = 'input.jpg'
img = cv2.imread(img_path)
original_height, original_width = img.shape[:2]

# prepare bbox and camera intrincis
bbox = [164, 93, 222, 252] # xmin, ymin, width, height
bbox = process_bbox(bbox, img.shape[1], img.shape[0])
assert len(bbox) == 4, 'Please set bbox'
focal = [1500, 1500] # x-axis, y-axis
assert len(focal) == 2, 'Please set focal length'
princpt = [img.shape[1]/2, img.shape[0]/2] # x-axis, y-axis
assert len(princpt) == 2, 'Please set princpt'
img, img2bb_trans = generate_patch_image(img, bbox, False, 0.0) 
img = transform(img).cuda()[None,:,:,:]
k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
k_value = torch.FloatTensor([k_value]).cuda()[None,:]

# forward
with torch.no_grad():
    root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
img = img[0].cpu().numpy()
root_3d = root_3d[0].cpu().numpy()

# save output in 2D space (x,y: pixel)
vis_img = img.copy()
vis_img = vis_img * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
vis_img = vis_img.astype(np.uint8)
vis_img = vis_img[::-1, :, :]
vis_img = np.transpose(vis_img,(1,2,0)).copy()
vis_root = np.zeros((2))
vis_root[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
vis_root[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
cv2.circle(vis_img, (vis_root[0], vis_root[1]), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
cv2.imwrite('output_root_2d.jpg', vis_img)

print('Depth from camera: ' + str(root_3d[2]) + ' mm') 

# camera back-projection
focal = (None, None) # focal length of x-axis, y-axis. please provide this. if do not know, set normalized one (1500, 1500).
princpt = (None, None) # princical point of x-axis, y-aixs. please provide this. if do not know, set (original_width/2, original_height/2).
# inverse affine transform (restore the crop and resize)
root_3d[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
root_3d[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
root_3d_xy1 = np.concatenate((root_3d[:2], np.ones_like(root_3d[:1])))
root_3d[:2] = np.dot(np.linalg.inv(img2bb_trans), root_3d_xy1)
root_3d = pixel2cam(root_3d, focal, princpt)
