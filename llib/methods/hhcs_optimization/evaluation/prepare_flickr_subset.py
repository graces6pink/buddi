"""
Pick a FlickrCI3D Signatures subset that can be run through the `demo` data
loader, so a trained BUDDI checkpoint can be evaluated on real photos without
the released `datasets/processed/FlickrCI3D_Signatures` bundle.

Why a subset, and why only single-pair images:

The normal FlickrCI3D route (llib/data/preprocess/flickrci3d_signatures_contacts.py)
hard-requires `processed.pkl`, which is built by process_flickrci3ds.py, which in
turn hard-requires a `correspondence.pkl` that no script in this repo produces
and that is anchored on OpenPose detections. Without the released bundle that
path is a dead end. The `demo` loader (llib/data/preprocess/demo.py) needs only
images + ViTPose + BEV, all of which we can produce locally.

The catch is identity: the demo loader enumerates person pairs by bbox IoU and
names its outputs `{imgname}_{pidx}` (demo.py:445), where pidx has no relation to
the annotation's `ci_sign` ordering -- and it skips a pair without incrementing
past it when BEV fails to match, so even the numbering is not contiguous. On an
image with exactly one annotated pair there is only one thing the result can
correspond to, so the mapping is unambiguous. That is 920 of the 1139 test
images, which is plenty.

Usage:
    python llib/methods/hhcs_optimization/evaluation/prepare_flickr_subset.py \
      --num-images 150 --out-folder demo/data/flickr_test_subset
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from loguru import logger as guru

from llib.utils.image.bbox import iou


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flickrci3ds-folder-orig', type=str,
                        default='datasets/original/FlickrCI3D_Signatures')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num-images', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-folder', type=str, default='demo/data/flickr_test_subset')
    parser.add_argument('--copy', action='store_true',
                        help='Copy images instead of symlinking them.')
    parser.add_argument('--filter-keypoints', action='store_true',
                        help='Second pass, run AFTER ViTPose: keep only the two '
                             'annotated people in each keypoints file. See '
                             'filter_keypoints() for why this is necessary.')
    parser.add_argument('--vitpose-folder', type=str, default='vitpose')
    parser.add_argument('--vitpose-out-folder', type=str, default='vitpose_pair')
    parser.add_argument('--min-bbox-iou', type=float, default=0.3)
    return parser.parse_args()


def bbox_from_keypoints(person):
    """Same bbox definition demo.py::bbox_from_openpose uses."""
    kpts = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
    valid = kpts[kpts[:, -1] > 0]
    if len(valid) == 0:
        return None
    x0, y0, _ = valid.min(0)
    x1, y1, _ = valid.max(0)
    return [x0, y0, x1, y1]


def filter_keypoints(args, imgnames, annotations):
    """Cut each keypoints file down to the two annotated people.

    Without this the evaluation is unaffordable. The demo loader pairs up every
    two detections whose boxes overlap (demo.py:425-433), and Flickr photos are
    crowded: 150 images produced 1199 pairs, ~8 per image, and the optimization
    costs ~60s per pair -- 20 hours per arm, for results that are 88% pairs no
    annotation refers to. Keeping only the annotated two leaves exactly one pair
    per image.

    It also fixes identity: with one pair per image the fit that comes back is
    unambiguously the pair the contact annotation describes.
    """
    in_folder = osp.join(args.out_folder, args.vitpose_folder)
    out_folder = osp.join(args.out_folder, args.vitpose_out_folder)
    os.makedirs(out_folder, exist_ok=True)

    kept, dropped = 0, []
    for imgname in imgnames:
        src = osp.join(in_folder, f'{imgname}_keypoints.json')
        if not osp.exists(src):
            dropped.append((imgname, 'no keypoints file'))
            continue

        data = json.load(open(src, 'r'))
        anno = annotations[imgname]['ci_sign'][0]
        gt_bboxes = [annotations[imgname]['bbxes'][pid] for pid in anno['person_ids']]

        det_bboxes = [bbox_from_keypoints(person) for person in data['people']]
        chosen = []
        for gt_bbox in gt_bboxes:
            scores = [iou(det, gt_bbox) if det is not None else -1.0
                      for det in det_bboxes]
            best = int(np.argmax(scores))
            chosen.append((best, scores[best]))

        if chosen[0][0] == chosen[1][0]:
            # both annotated people matched the same detection: the two were
            # detected as one person, the known failure mode on tight contact
            dropped.append((imgname, 'both annotated people matched one detection'))
            continue
        if min(score for _, score in chosen) < args.min_bbox_iou:
            dropped.append((imgname, f'weak bbox match {min(s for _, s in chosen):.2f}'))
            continue

        # keep annotation order, so person_ids[0] stays human 0
        data['people'] = [data['people'][idx] for idx, _ in chosen]
        with open(osp.join(out_folder, f'{imgname}_keypoints.json'), 'w') as f:
            json.dump(data, f)
        kept += 1

    guru.info(f'keypoint filter: kept {kept}/{len(imgnames)} images -> {out_folder}')
    for imgname, reason in dropped:
        guru.warning(f'  dropped {imgname}: {reason}')

    with open(osp.join(args.out_folder, 'keypoint_filter.json'), 'w') as f:
        json.dump({'kept': kept, 'dropped': dropped,
                   'min_bbox_iou': args.min_bbox_iou}, f, indent=2)


def main(args):
    split_folder = osp.join(args.flickrci3ds_folder_orig, args.split)
    image_folder = osp.join(split_folder, 'images')
    annotations = json.load(
        open(osp.join(split_folder, 'interaction_contact_signature.json'), 'r'))

    single_pair, multi_pair, missing = [], 0, 0
    for imgname, anno in annotations.items():
        if len(anno['ci_sign']) != 1:
            multi_pair += 1
            continue
        if not osp.exists(osp.join(image_folder, f'{imgname}.png')):
            missing += 1
            continue
        single_pair.append(imgname)
    single_pair.sort()

    guru.info(f'{len(annotations)} annotated images; {len(single_pair)} usable '
              f'(exactly one annotated pair), {multi_pair} skipped as multi-pair, '
              f'{missing} skipped as missing on disk')

    if args.filter_keypoints:
        manifest = json.load(open(osp.join(args.out_folder, 'subset.json'), 'r'))
        filter_keypoints(args, manifest['imgnames'], annotations)
        return

    n = min(args.num_images, len(single_pair))
    rng = np.random.RandomState(args.seed)
    chosen = sorted(rng.choice(single_pair, size=n, replace=False).tolist())

    out_images = osp.join(args.out_folder, 'images')
    os.makedirs(out_images, exist_ok=True)
    for imgname in chosen:
        src = osp.abspath(osp.join(image_folder, f'{imgname}.png'))
        dst = osp.join(out_images, f'{imgname}.png')
        if osp.lexists(dst):
            continue
        if args.copy:
            import shutil
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)

    manifest = osp.join(args.out_folder, 'subset.json')
    with open(manifest, 'w') as f:
        json.dump({'split': args.split, 'seed': args.seed, 'num_images': n,
                   'num_candidates': len(single_pair), 'imgnames': chosen}, f, indent=2)

    guru.info(f'Wrote {n} images to {out_images} and the manifest to {manifest}')


if __name__ == '__main__':
    main(parse_args())
