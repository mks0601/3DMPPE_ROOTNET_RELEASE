import os
import os.path as osp
import numpy as np
import json
from pycocotools.coco import COCO
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, process_bbox


class PW3D:
    def __init__(self, data_split):
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'PW3D', 'data')
        self.joint_num = 24
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
        self.root_idx = self.joints_name.index('Pelvis')
        self.joints_have_depth = True
        self.data = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)

            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(-1,3)
            joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
            joint_valid = ((joint_img[:,0] >= 0) * (joint_img[:,0] < img_width) * (joint_img[:,1] >= 0) * (joint_img[:,1] < img_height)).astype(np.float32)

            root_cam = joint_cam[self.root_idx]
            root_img = joint_img[self.root_idx]
            root_vis = joint_valid[self.root_idx]

            bbox = process_bbox(ann['bbox'], img_width, img_height)
            if bbox is None: continue
            area = bbox[2]*bbox[3]
            
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'ann_id': aid,
                'bbox': bbox,
                'area': area,
                'root_img': root_img,
                'root_cam': root_cam,
                'root_vis': root_vis,
                'f': cam_param['focal'],
                'c': cam_param['princpt']})
           
        return datalist

    def evaluate(self, preds, result_dir):
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
 
        pred_save = []
        errors = np.zeros((sample_num,3))
        for n in range(sample_num):
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']

            pred_root_coord = preds[n]
            pred_root_coord[0] = pred_root_coord[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_root_coord[1] = pred_root_coord[1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            pred_root_coord = pixel2cam(pred_root_coord[None,:], f, c)

            # error calculate
            pred_root_coord = pred_root_coord.reshape(3)
            gt_root_coord = gt['root_cam'].reshape(3)
            errors[n] = (pred_root_coord - gt_root_coord)**2

            # prediction save
            img_id = gt['img_id']
            ann_id = gt['ann_id']
            pred_root_coord = pred_root_coord.reshape(3)
            pred_save.append({'image_id': img_id, 'ann_id': ann_id, 'bbox': bbox.tolist(), 'root_cam': pred_root_coord.tolist()})
       
        err_x = np.mean(np.sqrt(errors[:,0]))
        err_y = np.mean(np.sqrt(errors[:,1]))
        err_z = np.mean(np.sqrt(errors[:,2]))
        err_total = np.mean(np.sqrt(np.sum(errors,1)))
        print('MRPE >> x: ' + str(err_x) + ' y: ' + str(err_y) + ' z: ' + str(err_z) + ' total: ' + str(err_total)) # error print (meter)

        output_path = osp.join(result_dir, 'rootnet_pw3d_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

