"""Step 1: run ViTPose over the FlickrCI3D test split, emitting all three
keypoint artifacts, then (separately) BEV.

Same trick as the CHI3D repro's s02_run_vitpose.py: the wholebody ViTPose model
already produces the raw 133-point COCO-WholeBody array, which is simultaneously
what the 'vitposeplus' channel indexes and, after the OpenPose-format conversion
in llib/utils/keypoints/vitpose_model.py:167-186, what the 'vitpose' and
'openpose' channels read. One forward pass, three files.

The folder layout differs from CHI3D -- keypoints live under keypoints/, bev does
not (process_flickrci3ds.py:161-164 and buddi_cond_bev.yaml:11-14):

    datasets/processed/FlickrCI3D_Signatures/test/
      bev/{imgname}_0.08.npz                     <- SINGLE underscore, see below
      keypoints/vitpose/{imgname}_keypoints.json
      keypoints/vitposeplus/{imgname}.pkl
      keypoints/openpose/{imgname}.json

DEVIATION: the OpenPose channel is a ViTPose copy. On Flickr this matters more
than it did on CHI3D, because OpenPose is not just one detector among four -- it
is the *index space* every other detector is matched into
(correspondence[img][method]['best_match'] is indexed by OpenPose person id,
correspondance.py:118-120) and it is the side that gets IoU-matched against the
annotated bboxes (process_flickrci3ds.py:182-184, 205). Substituting ViTPose is
self-consistent, since bbox_from_openpose only needs the 25-point layout that the
converted JSON already has, but it must be recorded.

BEV is NOT run here -- see --print-bev-cmd. Flickr wants {imgname}_0.08.npz with
a single underscore, while `bev -m video` produces the double-underscore
{imgname}__2_0.08.npz. Running `bev` per image (demo.sh:19-23 style) would give
the right name but reloads the model 1139 times. So: run video mode once, then
rename with f01b_rename_bev.py. The two modes call the identical `outputs =
bev(image)` per image and --temporal_optimize defaults to False, so no cross-frame
state exists and the rename is exact.
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
    return DefaultPredictor_Lazy(detectron2_cfg), ViTPoseModel(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder',
                    default='datasets/original/FlickrCI3D_Signatures')
    ap.add_argument('--processed-data-folder',
                    default='datasets/processed/FlickrCI3D_Signatures')
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--detectron-cfg', default=DETECTRON_CFG)
    ap.add_argument('--detectron-url', default=DETECTRON_URL)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--print-bev-cmd', action='store_true',
                    help='print the BEV command and exit, running nothing')
    args = ap.parse_args()

    image_dir = osp.join(args.original_data_folder, args.split, 'images')
    proc = osp.join(args.processed_data_folder, args.split)
    bev_dir = osp.join(proc, 'bev')

    if args.print_bev_cmd:
        print(f'bev -m video -i {image_dir} -o {bev_dir}')
        print(f'python datasets/scripts/FlickrCI3D/repro/f01b_rename_bev.py')
        return

    vitpose_dir = osp.join(proc, 'keypoints', 'vitpose')
    vitposeplus_dir = osp.join(proc, 'keypoints', 'vitposeplus')
    openpose_dir = osp.join(proc, 'keypoints', 'openpose')
    for d in (vitpose_dir, vitposeplus_dir, openpose_dir):
        os.makedirs(d, exist_ok=True)

    img_fns = sorted(os.listdir(image_dir))
    print(f'{len(img_fns)} images in {image_dir}')

    detector, cpm = build_models(args.device, args.detectron_cfg, args.detectron_url)

    n_people, failed = [], []
    for img_fn in tqdm(img_fns, desc=args.split):
        name = osp.splitext(img_fn)[0]
        vitpose_fn = osp.join(vitpose_dir, f'{name}_keypoints.json')
        if osp.exists(vitpose_fn) and not args.overwrite:
            continue

        img_cv2 = cv2.imread(osp.join(image_dir, img_fn))
        if img_cv2 is None:
            failed.append((name, 'unreadable image'))
            continue

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
        # process_flickrci3ds.py:162 wants {imgname}.json with no _keypoints suffix
        with open(osp.join(openpose_dir, f'{name}.json'), 'w') as f:
            json.dump(contents, f)
        with open(osp.join(vitposeplus_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(vitposes_out, f)

    if n_people:
        counts = np.bincount(n_people)
        print('detections per image -> '
              + ', '.join(f'{n}: {c}' for n, c in enumerate(counts) if c))
        # every annotated pair needs two people; correspondence drops the rest
        lt2 = counts[:2].sum() if len(counts) > 1 else counts.sum()
        print(f'images with fewer than 2 detected people: {lt2}')
    for f in failed:
        print(f'  FAILED {f}')
    print(f'{len(os.listdir(vitpose_dir))} vitpose, '
          f'{len(os.listdir(vitposeplus_dir))} vitposeplus, '
          f'{len(os.listdir(openpose_dir))} openpose files')


if __name__ == '__main__':
    main()
