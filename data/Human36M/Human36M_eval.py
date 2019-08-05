import os
import os.path as osp
import json
import numpy as np
from pycocotools.coco import COCO

def calculate_score(output_path, annot_dir, subject_list):
    joint_num = 17 
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    
    # aggregate annotations from each subject
    db = COCO()
    for subject in subject_list:
        with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '.json'),'r') as f:
            annot = json.load(f)
        if len(db.dataset) == 0:
            for k,v in annot.items():
                db.dataset[k] = v
        else:
            for k,v in annot.items():
                db.dataset[k] += v
    db.createIndex()
        
    with open(output_path,'r') as f:
        output = json.load(f)

    error = np.zeros((len(output), 1, 3)) # MRPE
    error_action = [ [] for _ in range(len(action_name)) ] # MRPE for each action
    for n in range(len(output)):
        
        img_id = output[n]['image_id']
        root_cam_out = np.array(output[n]['root_cam'])
        
        gt_ann_id = db.getAnnIds(imgIds=[img_id])
        gt_ann = db.loadAnns(gt_ann_id)[0]
        root_idx = 0
        root_cam_gt = np.array(gt_ann['keypoints_cam'][root_idx])
        
        error[n] = np.power(root_cam_out - root_cam_gt,2)
        img = db.loadImgs([img_id])[0]
        img_name = img['file_name']
        action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
        error_action[action_idx].append(error[n].copy())

    # total error
    tot_err = np.mean(np.power(np.sum(error,axis=2),0.5))
    x_err = np.mean(np.power(error[:,:,0],0.5))
    y_err = np.mean(np.power(error[:,:,1],0.5))
    z_err = np.mean(np.power(error[:,:,2],0.5))
    eval_summary = 'MRPE >> tot: %.2f, x: %.2f, y: %.2f, z: %.2f\n' % (tot_err, x_err, y_err, z_err)
   
    # error for each action
    for i in range(len(error_action)):
        err = np.array(error_action[i])
        err = np.mean(np.power(np.sum(err,axis=2),0.5))
        eval_summary += (action_name[i] + ': %.2f ' % err)
    print(eval_summary)

if __name__ == '__main__':
    output_path = './bbox_root_human36m_output.json'
    annot_dir = './data/annotations'
    calculate_score(output_path, annot_dir, [9,11])

