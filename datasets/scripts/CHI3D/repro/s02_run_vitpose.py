"""Step 2a: run ViTPose once per frame and emit all three keypoint artifacts.

process_chi3d.py reads three separate keypoint sources and, unlike BEV, has no
os.path.exists guard on any of them (process_chi3d.py:233-235) -- a missing file
is a hard FileNotFoundError. On top of that chi3d_items_for_eval
(evaluation/utils.py:44-50) drops any sample where *any* of the four
*_human_idx is -1, so an empty OpenPose channel silently reduces Table 3 to zero
evaluated samples.

We do not have OpenPose installed. The observation that makes this cheap: the
wholebody ViTPose model already produces the raw 133-keypoint COCO-WholeBody
array, and llib/utils/keypoints/vitpose_model.py:167-186 converts it to exactly
the OpenPose-format JSON that both the 'vitpose' and 'openpose' channels expect,
while correspondance.py:95-106 / process_chi3d.py:308-318 index the *raw* 133
array for the 'vitposeplus' channel. So one forward pass yields all three:

    images_contact_vitpose/{name}_keypoints.json      OpenPose-format JSON
    images_contact_vitposeplus/{name}.pkl             raw list of {'keypoints': (133,3)}
    images_contact_openpose/keypoints/{name}.json     copy of the vitpose JSON

DEVIATION: the OpenPose channel is a ViTPose copy, not real OpenPose. With
use_hands=False (what buddi_cond_bev.yaml sets) single_optimization.py:225-236
uses vitpose as the fitting target and touches openpose only to (a) substitute
for humans vitpose missed and (b) copy two toe keypoints when the ankle residual
is < 5.0. When the two sources are identical (a) never fires and (b) is an
identity operation, so the effect on the fit is nil -- its real job is to keep
openpose_human_idx != -1. Record this when reporting numbers.
"""

import argparse
import json
import os
import os.path as osp
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from llib.utils.keypoints.vitpose_model import ViTPoseModel, DETECTRON_CFG, DETECTRON_URL


def to_openpose_json(vitposes_out):
    """Verbatim port of vitpose_model.py:167-186 so the JSON is byte-compatible."""
    json_contents = {'people': []}
    for person in vitposes_out:
        keypoints = person['keypoints'].astype(np.float64)
        keypoints_body = keypoints[:17, :]
        keypoints_left_hand = keypoints[-42:-21, :].reshape(-1)
        keypoints_right_hand = keypoints[-21:].reshape(-1)
        keypoints_face = np.concatenate((keypoints[23:-42].reshape(-1), np.zeros(6)))
        body_pose = np.zeros([25, 3])
        body_pose[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = keypoints_body
        body_pose[[19, 20, 21]] = keypoints[[17, 18, 19], :]  # left foot
        body_pose[[22, 23, 24]] = keypoints[[20, 21, 22], :]  # right foot
        body_pose = np.reshape(body_pose, -1)

        json_contents['people'].append({
            'pose_keypoints_2d': list(body_pose),
            'hand_left_keypoints_2d': list(keypoints_left_hand),
            'hand_right_keypoints_2d': list(keypoints_right_hand),
            'face_keypoints_2d': list(keypoints_face),
        })
    return json_contents


def build_models(device, detectron_cfg, detectron_url):
    from llib.utils.keypoints.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig

    detectron2_cfg = LazyConfig.load(str(Path(detectron_cfg)))
    detectron2_cfg.train.init_checkpoint = detectron_url
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(device)
    return detector, cpm


def run_subject(proc_root, split, subject, detector, cpm, overwrite=False):
    subj_dir = osp.join(proc_root, split, subject)
    image_dir = osp.join(subj_dir, 'images_contact')
    vitpose_dir = osp.join(subj_dir, 'images_contact_vitpose')
    vitposeplus_dir = osp.join(subj_dir, 'images_contact_vitposeplus')
    openpose_dir = osp.join(subj_dir, 'images_contact_openpose', 'keypoints')
    for d in (vitpose_dir, vitposeplus_dir, openpose_dir):
        os.makedirs(d, exist_ok=True)

    img_fns = sorted(os.listdir(image_dir))
    n_people = []
    for img_fn in tqdm(img_fns, desc=subject):
        name = osp.splitext(img_fn)[0]
        vitpose_fn = osp.join(vitpose_dir, f'{name}_keypoints.json')
        if osp.exists(vitpose_fn) and not overwrite:
            continue

        img_cv2 = cv2.imread(osp.join(image_dir, img_fn))
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid].cpu().numpy()
        scores = det_instances.scores[valid].cpu().numpy()

        vitposes_out = cpm.predict_pose(
            img_cv2, [np.concatenate([boxes, scores[:, None]], axis=1)])
        n_people.append(len(vitposes_out))

        contents = to_openpose_json(vitposes_out)
        with open(vitpose_fn, 'w') as f:
            json.dump(contents, f)
        # the 'openpose' channel is the same detections under the name
        # process_chi3d.py:205 expects ({name}.json, no _keypoints suffix)
        with open(osp.join(openpose_dir, f'{name}.json'), 'w') as f:
            json.dump(contents, f)
        # raw 133-keypoint arrays for the 'vitposeplus' channel
        with open(osp.join(vitposeplus_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(vitposes_out, f)

    if n_people:
        counts = np.bincount(n_people)
        print(f'{subject}: detections per image -> '
              + ', '.join(f'{n} people: {c}' for n, c in enumerate(counts) if c))
        if len(counts) > 2 and counts[2:].sum() < 0.9 * len(n_people):
            print(f'{subject}: WARNING only {counts[2:].sum()}/{len(n_people)} images '
                  'have >=2 detected people; correspondence will drop the rest')
    print(f'{subject}: {len(os.listdir(vitpose_dir))} vitpose, '
          f'{len(os.listdir(vitposeplus_dir))} vitposeplus, '
          f'{len(os.listdir(openpose_dir))} openpose files')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--split', default='train')
    ap.add_argument('--subjects', nargs='+', default=['s03'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--detectron-cfg', default=DETECTRON_CFG)
    ap.add_argument('--detectron-url', default=DETECTRON_URL)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    detector, cpm = build_models(args.device, args.detectron_cfg, args.detectron_url)
    for subject in args.subjects:
        run_subject(args.processed_data_folder, args.split, subject,
                    detector, cpm, args.overwrite)


if __name__ == '__main__':
    main()
