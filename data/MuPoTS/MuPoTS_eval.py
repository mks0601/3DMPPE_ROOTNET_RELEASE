import json
from pycocotools.coco import COCO
import os
import os.path as osp
import numpy as np
import math

def calculate_score(output_path, annot_path, thr):

    with open(output_path, 'r') as f:
        output = json.load(f)

    # AP measure
    def return_score(pred):
        return pred['score']
    output.sort(reverse=True, key=return_score)

    db = COCO(annot_path)
    gt_num = len([k for k,v in db.anns.items() if v['is_valid'] == 1])
    tp_acc = 0
    fp_acc = 0
    precision = []; recall = [];
    is_matched = {}
    for n in range(len(output)):
        image_id = output[n]['image_id']
        pred_root = output[n]['root_cam']
        score = output[n]['score']

        img = db.loadImgs(image_id)[0]
        ann_ids = db.getAnnIds(image_id)
        anns = db.loadAnns(ann_ids)
        valid_frame_num = len([item for item in anns if item['is_valid'] == 1])
        if valid_frame_num == 0:
            continue

        if str(image_id) not in is_matched:
            is_matched[str(image_id)] = [0 for _ in range(len(anns))]
        
        min_dist = 9999
        save_ann_id = -1
        for ann_id,ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            root_idx = 14
            gt_root = gt_root[root_idx]

            dist = math.sqrt(np.sum((pred_root - gt_root) ** 2))
            if min_dist > dist:
                min_dist = dist
                save_ann_id = ann_id
        
        is_tp = False
        if save_ann_id != -1 and min_dist < thr:
            if is_matched[str(image_id)][save_ann_id] == 0:
                is_tp = True
                is_matched[str(image_id)][save_ann_id] = 1
        
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
            
        precision.append(tp_acc/(tp_acc + fp_acc))
        recall.append(tp_acc/gt_num)

    AP = 0
    for n in range(len(precision)-1):
        AP += precision[n+1] * (recall[n+1] - recall[n])

    print('AP_root: ' + str(AP))

if __name__ == '__main__':
    output_path = './bbox_root_mupots_output.json'
    annot_path = osp.join('..', '..', 'data', 'MuPoTS', 'data', 'MuPoTS-3D.json')
    thr = 250
    calculate_score(output_path, annot_path, thr)

