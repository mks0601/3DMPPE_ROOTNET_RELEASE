import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, get_bbox
import cv2
import random
import json
import math
from Human36M_eval import calculate_score

class Human36M:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'Human36M', 'images')
        self.annot_dir = osp.join('..', 'data', 'Human36M', 'annotations')
        self.human_bbox_dir = osp.join('..', 'data', 'Human36M', 'bbox', 'bbox_human36m_output.json')
        self.joint_num = 18
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        self.root_idx = self.joints_name.index('Pelvis')
        self.min_depth = 2000
        self.max_depth = 8000
        self.joints_have_depth = True
        self.protocol = 1
        self.subject_list = self.get_subject()
        self.sampling_ratio = self.get_subsampling_ratio()
        self.data = self.load_data()
       
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):

        # aggregate annotations from each subject
        db = COCO()
        for subject in self.subject_list:
            with open(osp.join(self.annot_dir, 'Human36M_subject' + str(subject) + '.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
        db.createIndex()

        if self.data_split == 'test' and not cfg.use_gt_bbox:
            print("Get bounding box from " + self.human_bbox_dir)
            bbox_result = {}
            with open(self.human_bbox_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_result[str(annot[i]['image_id'])] = np.array(annot[i]['bbox'])
        else:
            print("Get bounding box from groundtruth")

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            
            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in self.subject_list:
                continue
            if frame_idx % self.sampling_ratio != 0:
                continue

            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            cam_param = img['cam_param']
            R,t,f,c = np.array(cam_param['R']), np.array(cam_param['t']), np.array(cam_param['f']), np.array(cam_param['c'])
                
            # project world coordinate to cam, image coordinate space
            root_cam = np.array(ann['keypoints_cam'])[self.root_idx]
            root_img = np.array(ann['keypoints_img'])[self.root_idx]
            root_img = np.concatenate([root_img, root_cam[2:3]])
            root_vis = np.array(ann['keypoints_vis'])[self.root_idx,None]

            if self.data_split == 'test' and not cfg.use_gt_bbox:
                bbox = bbox_result[str(image_id)]
            else:
                bbox = np.array(ann['bbox'])

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
                'img_path': img_path,
                'img_id': image_id,
                'bbox': bbox,
                'area': area,
                'root_img': root_img, # [org_img_x, org_img_y, depth]
                'root_cam': root_cam,
                'root_vis': root_vis,
                'f': f,
                'c': c
            })

        return data

    def evaluate(self, preds, result_dir):
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
 
        pred_save = []
        for n in range(sample_num):
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_root = gt['root_cam']
            
            # warp output to original image space
            pred_root = preds[n]
            pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            
            # back-project to camera coordinate space
            pred_root[0], pred_root[1], pred_root[2] = pixel2cam(pred_root, f, c)

            # prediction save
            img_id = gt['img_id']
            pred_save.append({'image_id': img_id, 'bbox': bbox.tolist(), 'root_cam': pred_root.tolist()})
        
        output_path = osp.join(result_dir, 'bbox_root_human36m_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

        calculate_score(output_path, self.annot_dir, self.subject_list)

