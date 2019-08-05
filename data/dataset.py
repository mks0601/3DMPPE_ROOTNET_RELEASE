import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from config import cfg

class DatasetLoader(Dataset):
    def __init__(self, db, is_train, transform):
        
        self.db = db.data
        self.joint_num = db.joint_num
        self.root_idx = db.root_idx
        self.joints_have_depth = db.joints_have_depth
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

    def __getitem__(self, index):
        
        joints_have_depth = self.joints_have_depth 
        data = copy.deepcopy(self.db[index])

        bbox = data['bbox']
        root_img = np.array(data['root_img'])
        root_vis = np.array(data['root_vis'])
        area = data['area']
        f = data['f']

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape
        
        # 2. get augmentation params
        if self.do_augment:
            rot, do_flip, color_scale = get_aug_config()
        else:
            rot, do_flip, color_scale = 0, False, [1.0, 1.0, 1.0]

        # 3. crop patch from img and perform data augmentation (flip, rot, color scale)
        img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, rot)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

        # 4. generate patch joint, area_ratio, and ground truth
        # flip joints and apply Affine Transform on joints
        if do_flip:
            root_img[0] = img_width - root_img[0] - 1
        root_img[0:2] = trans_point2d(root_img[0:2], trans)
        root_vis *= (
                        (root_img[0] >= 0) & \
                        (root_img[0] < cfg.input_shape[1]) & \
                        (root_img[1] >= 0) & \
                        (root_img[1] < cfg.input_shape[0])
                        )
        
        # change coordinates to output space
        root_img[0] = root_img[0] / cfg.input_shape[1] * cfg.output_shape[1]
        root_img[1] = root_img[1] / cfg.input_shape[0] * cfg.output_shape[0]
        
        if self.is_train:
            img_patch = self.transform(img_patch)
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)
            root_img = root_img.astype(np.float32)
            root_vis = root_vis.astype(np.float32)
            joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

            return img_patch, k_value, root_img, root_vis, joints_have_depth
        else:
            img_patch = self.transform(img_patch)
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)
          
            return img_patch, k_value

    def __len__(self):
        return len(self.db)

# helper functions
def get_aug_config():
   
    rot_factor = 30
    color_factor = 0.2
    
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return rot, do_flip, color_scale

def generate_patch_image(cvimg, bbox, do_flip, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, rot, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

