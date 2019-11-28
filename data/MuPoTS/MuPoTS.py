import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
from config import cfg
import json
import cv2
import random
import math
from utils.pose_utils import pixel2cam, process_bbox
from sklearn.metrics import average_precision_score
from MuPoTS_eval import calculate_score

class MuPoTS:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MuPoTS', 'data', 'MultiPersonTestSet')
        self.annot_path = osp.join('..', 'data', 'MuPoTS', 'data', 'MuPoTS-3D.json')
        self.human_bbox_dir = osp.join('..', 'data', 'MuPoTS', 'bbox', 'bbox_mupots_output.json')
        self.joint_num = 21 # MuCo-3DHP
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') # MuCo-3DHP
        self.original_joint_num = 17 # MuPoTS
        self.original_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head') # MuPoTS

        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):
        
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0
        
        data = []
        db = COCO(self.annot_path)
        if cfg.use_gt_bbox:
            print("Get bounding box from groundtruth")

            for aid in db.anns.keys():
                ann = db.anns[aid]
                if ann['is_valid'] == 0:
                    continue

                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);

                joint_cam = np.array(ann['keypoints_cam'])
                joint_img = np.array(ann['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_vis = np.array(ann['keypoints_vis'])

                root_cam = joint_cam[self.root_idx]
                root_img = joint_img[self.root_idx]
                root_vis = joint_vis[self.root_idx,None]
 
                bbox = np.array(ann['bbox'])
                img_width, img_height = img['width'], img['height']
                bbox = process_bbox(bbox, img_width, img_height)
                if bbox is None: continue
                area = bbox[2]*bbox[3]

                data.append({
                    'image_id': ann['image_id'],
                    'img_path': img_path,
                    'bbox': bbox,
                    'area': area,
                    'root_img': root_img, # [org_img_x, org_img_y, depth - root_depth]
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'root_vis': root_vis,
                    'f': f,
                    'c': c,
                    'score': 1.0
                })
        else:
            with open(self.human_bbox_dir) as f:
                annot = json.load(f)
            print("Get bounding box from " + self.human_bbox_dir)

            for i in range(len(annot)):
                image_id = annot[i]['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);

                bbox = np.array(annot[i]['bbox']).reshape(4)
                img_width, img_height = img['width'], img['height']
                bbox = process_bbox(bbox, img_width, img_height)
                if bbox is None: continue
                area = bbox[2]*bbox[3]

                data.append({
                    'image_id': image_id,
                    'img_path': img_path,
                    'bbox': bbox,
                    'area': area,
                    'root_img': np.ones((3)), # dummy
                    'root_cam': np.ones((3)), # dummy
                    'root_vis': np.ones((1)), # dummy
                    'f': f,
                    'c': c,
                    'score': annot[i]['score']
                })
        return data

    def evaluate(self, preds, result_dir):
        
        print('Evaluation start...')
        pred_save = []

        gts = self.data
        sample_num = len(preds)
        for n in range(sample_num):
            
            gt = gts[n]
            image_id = gt['image_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox'].tolist()
            score = gt['score']
            
            # restore coordinates to original space
            pred_root = preds[n].copy()
            pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]

            # back project to camera coordinate system
            pred_root = pixel2cam(pred_root[None,:], f, c)[0]

            pred_save.append({'image_id': image_id, 'root_cam': pred_root.tolist(), 'bbox': bbox, 'score': score})
        
        output_path = osp.join(result_dir, 'bbox_root_mupots_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

        calculate_score(output_path, self.annot_path, 250)

 

