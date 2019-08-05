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
from utils.pose_utils import pixel2cam, get_bbox
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
        self.min_depth = 1500
        self.max_depth = 7500
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

                # sanitize bboxes
                x, y, w, h = bbox
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
                if w*h > 0 and x2 >= x1 and y2 >= y1:
                    bbox = np.array([x1, y1, x2-x1, y2-y1])
                else:
                    continue

                # aspect ratio preserving bbox
                w = bbox[2]
                h = bbox[3]
                c_x = bbox[0] + w/2.
                c_y = bbox[1] + h/2.
                aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
                if w > aspect_ratio * h:
                    h = w / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio
                bbox[2] = w*1.25
                bbox[3] = h*1.25
                bbox[0] = c_x - bbox[2]/2.
                bbox[1] = c_y - bbox[3]/2.
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

                # sanitize bboxes
                x, y, w, h = bbox
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
                if w*h > 0 and x2 >= x1 and y2 >= y1:
                    bbox = np.array([x1, y1, x2-x1, y2-y1])
                else:
                    continue

                # aspect ratio preserving bbox
                w = bbox[2]
                h = bbox[3]
                c_x = bbox[0] + w/2.
                c_y = bbox[1] + h/2.
                aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
                if w > aspect_ratio * h:
                    h = w / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio
                bbox[2] = w*1.25
                bbox[3] = h*1.25
                bbox[0] = c_x - bbox[2]/2.
                bbox[1] = c_y - bbox[3]/2.
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
            pred_root[0], pred_root[1], pred_root[2] = pixel2cam(pred_root, f, c)
            pred_root = pred_root.tolist()

            pred_save.append({'image_id': image_id, 'root_cam': pred_root, 'bbox': bbox, 'score': score})
        
        output_path = osp.join(result_dir, 'bbox_root_mupots_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

        calculate_score(output_path, self.annot_path, 250)

 

