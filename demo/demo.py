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
        assert 0, "Please set proper gpu ids"

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
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox for each human
bbox_list = [
[139.41, 102.25, 222.39, 241.57],\
[287.17, 61.52, 74.88, 165.61],\
[540.04, 48.81, 99.96, 223.36],\
[372.58, 170.84, 266.63, 217.19],\
[0.5, 43.74, 90.1, 220.09]] # xmin, ymin, width, height
person_num = len(bbox_list)

# normalized camera intrinsics
focal = [1500, 1500] # x-axis, y-axis
princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

# for cropped and resized human image, forward it to RootNet
for n in range(person_num):
    bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
    img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0) 
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
    cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.imwrite('output_root_2d_' + str(n) + '.jpg', vis_img)
    
    print('Root joint depth: ' + str(root_3d[2]) + ' mm')

