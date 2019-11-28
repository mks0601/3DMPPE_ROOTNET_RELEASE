import os
import os.path as osp
import numpy as np
from utils.pose_utils import process_bbox
from pycocotools.coco import COCO
from config import cfg
import math

class MuCo:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MuCo', 'data')
        self.train_annot_path = osp.join('..', 'data', 'MuCo', 'data', 'MuCo-3DHP.json')
        self.joint_num = 21
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):

        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
        else:
            print('Unknown data subset')
            assert 0

        data = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            f = img["f"]
            c = img["c"]

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)
            
            # exclude too close persons
            root_depths = [ann['keypoints_cam'][self.root_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]
            for i in range(len(anns)):
                if i == closest_pid:
                    continue
                picked = True
                for j in range(len(anns)):
                    if i == j:
                        continue
                    dist = (np.array(anns[i]['keypoints_cam'][self.root_idx]) - np.array(anns[j]['keypoints_cam'][self.root_idx])) ** 2
                    dist_2d = math.sqrt(np.sum(dist[:2]))
                    dist_3d = math.sqrt(np.sum(dist))
                    if dist_2d < 500 or dist_3d < 500:
                        picked = False
                if picked:
                    pid_list.append(i)
            
            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                root_cam = joint_cam[self.root_idx]

                if root_cam[2] < self.min_depth or root_cam[2] > self.max_depth:
                    continue
                
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                root_img = joint_img[self.root_idx]
                
                joint_vis = np.array(anns[pid]['keypoints_vis'])
                root_vis = joint_vis[self.root_idx,None]
                bbox = process_bbox(anns[pid]['bbox'], img_width, img_height)
                if bbox is None: continue
                area = bbox[2]*bbox[3]

                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'area': area,
                    'root_img': root_img, # [org_img_x, org_img_y, depth]
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'root_vis': root_vis,
                    'f': f,
                    'c': c
                })
        return data


