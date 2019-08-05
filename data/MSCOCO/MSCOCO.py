import os
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
from config import cfg
from utils.pose_utils import pixel2cam
import json

class MSCOCO:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MSCOCO', 'images')
        self.train_annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations', 'person_keypoints_train2017.json')
        self.test_annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations', 'person_keypoints_val2017.json')
        self.human_bbox_dir = osp.join('..', 'data', 'MSCOCO', 'bbox_coco_output.json')
        self.joint_num = 19 # original: 17, but manually added 'Thorax', 'Pelvis'
        self.joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
        self.joints_have_depth = False
        self.min_depth = 0 #dummy
        self.max_depth = 1 #dummy

        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.lhip_idx = self.joints_name.index('L_Hip')
        self.rhip_idx = self.joints_name.index('R_Hip')
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):

        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
            data = []
            for aid in db.anns.keys():
                ann = db.anns[aid]

                if (ann['image_id'] not in db.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # sanitize bboxes
                x, y, w, h = ann['bbox']
                img = db.loadImgs(ann['image_id'])[0]
                width, height = img['width'], img['height']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
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

                # joints and vis
                joint_img = np.array(ann['keypoints']).reshape(-1,3)
                # add Thorax
                thorax = (joint_img[self.lshoulder_idx, :] + joint_img[self.rshoulder_idx, :]) * 0.5
                thorax[2] = joint_img[self.lshoulder_idx,2] * joint_img[self.rshoulder_idx,2]
                thorax = thorax.reshape((1, 3))
                # add Pelvis
                pelvis = (joint_img[self.lhip_idx, :] + joint_img[self.rhip_idx, :]) * 0.5
                pelvis[2] = joint_img[self.lhip_idx,2] * joint_img[self.rhip_idx,2]
                pelvis = pelvis.reshape((1, 3))

                joint_img = np.concatenate((joint_img, thorax, pelvis), axis=0)

                joint_vis = (joint_img[:,2].copy().reshape(-1,1) > 0)
                joint_img[:,2] = 0

                root_img = joint_img[self.root_idx]
                root_vis = joint_vis[self.root_idx]

                imgname = osp.join('train2017', db.imgs[ann['image_id']]['file_name'])
                img_path = osp.join(self.img_dir, imgname)
                data.append({
                    'img_path': img_path,
                    'image_id': ann['image_id'],
                    'bbox': bbox,
                    'area': area,
                    'root_img': root_img, # [org_img_x, org_img_y, 0]
                    'root_vis': root_vis,
                    'f': np.array([1500, 1500]) # dummy value
                })

        elif self.data_split == 'test':
            db = COCO(self.test_annot_path)
            data = []
            for aid in db.anns.keys():
                ann = db.anns[aid]

                if (ann['image_id'] not in db.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # sanitize bboxes
                x, y, w, h = ann['bbox']
                img = db.loadImgs(ann['image_id'])[0]
                width, height = img['width'], img['height']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
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

                # joints and vis
                joint_img = np.array(ann['keypoints']).reshape(-1,3)
                # add Thorax
                thorax = (joint_img[self.lshoulder_idx, :] + joint_img[self.rshoulder_idx, :]) * 0.5
                thorax[2] = joint_img[self.lshoulder_idx,2] * joint_img[self.rshoulder_idx,2]
                thorax = thorax.reshape((1, 3))
                # add Pelvis
                pelvis = (joint_img[self.lhip_idx, :] + joint_img[self.rhip_idx, :]) * 0.5
                pelvis[2] = joint_img[self.lhip_idx,2] * joint_img[self.rhip_idx,2]
                pelvis = pelvis.reshape((1, 3))

                joint_img = np.concatenate((joint_img, thorax, pelvis), axis=0)

                joint_vis = (joint_img[:,2].copy().reshape(-1,1) > 0)
                joint_img[:,2] = 0

                root_img = joint_img[self.root_idx]
                root_vis = joint_vis[self.root_idx]

                imgname = osp.join('val2017', db.imgs[ann['image_id']]['file_name'])
                img_path = osp.join(self.img_dir, imgname)
                data.append({
                    'img_path': img_path,
                    'image_id': ann['image_id'],
                    'bbox': bbox,
                    'area': area,
                    'root_img': root_img, # [org_img_x, org_img_y, 0]
                    'root_vis': root_vis,
                    'f': np.array([1500, 1500]), # dummy value
                    'c': np.array([width/2, height/2]) # dummy value
                })

        else:
            print('Unknown data subset')
            assert 0

        return data

    def evaluate(self, preds, result_dir):
        
        print('Evaluation start...')
        gts = self.data
        sample_num = len(preds)
        pred_save = []
        for n in range(sample_num):
            
            gt = gts[n]
            image_id = gt['image_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox'].tolist()
            
            # restore coordinates to original space
            pred_root = preds[n].copy()
            pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]

            # back project to camera coordinate system
            pred_root[0], pred_root[1], pred_root[2] = pixel2cam(pred_root, f, c)
            pred_root = pred_root.tolist()

            pred_save.append({'image_id': image_id, 'root_cam': pred_root, 'bbox': bbox})
        
        output_path = osp.join(result_dir, 'bbox_root_coco_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Testing result is saved at " + output_path)
