"""
Contact-only evaluation of optimization runs on FlickrCI3D Signatures.

This is the contact track of flickrci3ds_eval.py, sliced out so it can run
without the released pseudo-ground-truth fits.

flickrci3ds_eval.py evaluates on two tracks: contact, against the human
annotations in interaction_contact_signature.json (real GT), and joint error,
against pseudo-GT fits produced by stage (1) optimization. The pseudo-GT is only
available from the paper's Google Drive bundle, and it cannot be regenerated
locally: the stage (1) loader needs datasets/processed/.../processed.pkl, which
needs a correspondence.pkl that no script here produces and that is anchored on
OpenPose. So the joint-error track is out of reach without that download.

The contact track is not. flickrci3ds_eval.py:249-263 needs exactly two things:
the predicted vertices and the annotation JSON. Both are available. That is what
this script computes, for several runs at once, on the same images:

  IoU / precision / recall / F1 of the 13mm binary contact map (same threshold
  and same ContactMap/ContactIOU code as flickrci3ds_eval.py), PCC, the mean
  distance between the region pairs the annotation says are touching, and --
  needing no ground truth at all -- mesh interpenetration depth.

Coverage matters as much as the scores: a method that silently fails on hard
images would otherwise look good by averaging over the easy ones. Every metric
is reported twice, once over each run's own results and once over the subset
where *every* run produced a result, which is the only fair comparison.

Usage:
    python llib/methods/hhcs_optimization/evaluation/flickrci3ds_contact_eval.py \
      --runs baseline-nodiff=demo/optimization/flickr_contact_eval/baseline-nodiff \
             buddi-official=demo/optimization/flickr_contact_eval/buddi-official \
             buddi-mine=demo/optimization/flickr_contact_eval/buddi-mine \
      --output-folder demo/optimization/flickr_contact_eval/comparison
"""

import argparse
import csv
import glob
import json
import os
import os.path as osp
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from loguru import logger as guru
from tqdm import tqdm

from llib.bodymodels.build import build_bodymodel
from llib.cameras.build import build_camera
from llib.data.preprocess.utils.check_interx_penetration import score_batch
from llib.defaults.main import config as default_config, merge as merge_configs
from llib.utils.image.bbox import iou as bbox_iou
from llib.utils.metrics.contact import ContactIOU, MaxIntersection
from llib.utils.threed.distance import ContactMap

REGION_TO_VERTEX_PATH = 'essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl'
NUM_REGIONS = 75


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp-cfg', dest='exp_cfgs', nargs='+',
                        default=['llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.yaml'],
                        help='Body model / camera config. The default is the one '
                             'flickrci3ds_eval.py uses.')
    parser.add_argument('--exp-opts', dest='exp_opts', nargs='*', default=[])
    parser.add_argument('--runs', nargs='+', required=True,
                        help='name=path pairs. Each path is an optimization run '
                             'folder holding results/ (or pkl/) with '
                             '{imgname}_{pair}.pkl files.')
    parser.add_argument('--annotations', type=str,
                        default='datasets/original/FlickrCI3D_Signatures/test/'
                                'interaction_contact_signature.json')
    parser.add_argument('--subset', type=str,
                        default='demo/data/flickr_test_subset/subset.json',
                        help='Manifest from prepare_flickr_subset.py. Pass "none" '
                             'to evaluate every single-pair annotated image.')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--contact-threshold', type=float, default=0.013)
    parser.add_argument('--penetration-lowres', type=int, default=500,
                        help='Downsampled face count for the interpenetration '
                             'test. 500 is the value calibrated in '
                             'check_interx_penetration.py.')
    parser.add_argument('--min-bbox-iou', type=float, default=0.3,
                        help='Reject a fit whose reprojected pair does not overlap '
                             'the annotated pair by at least this mean IoU.')
    parser.add_argument('--images-folder', type=str,
                        default='datasets/original/FlickrCI3D_Signatures/test/images')
    parser.add_argument('--skip-penetration', action='store_true')

    cmd_args = parser.parse_args()
    cfg = merge_configs(cmd_args, default_config)
    return cfg, cmd_args


def parse_runs(run_args):
    runs = {}
    for entry in run_args:
        assert '=' in entry, f'--runs takes name=path pairs, got {entry!r}'
        name, path = entry.split('=', 1)
        runs[name] = path
    return runs


def candidate_results(run_folder, imgname):
    """All result pkls the demo loader wrote for this image.

    The demo loader enumerates every bbox-overlapping person pair and names the
    outputs {imgname}_{pidx} (demo.py:425-445). An image with a single annotated
    pair can still yield several of these -- a third detected person overlapping
    one of the two is enough -- so the file index says nothing about which pair
    the annotation means. match_result() picks by geometry instead.
    """
    matches = []
    for sub in ['results', 'pkl']:
        matches += sorted(glob.glob(osp.join(run_folder, sub, f'{imgname}_*.pkl')))
    return matches


def load_result(result_path, body_model, camera, image_size, device):
    """Result pkl -> (vertices [2,V,3], projected 2D bboxes [2,4]).

    Fields are the ones flickrci3ds_eval.py::get_smplx_params reads, plus the
    camera the optimization converged to, which is what lets us put the fit back
    into image space and check which annotated pair it corresponds to.
    """
    item = pickle.load(open(result_path, 'rb'))
    humans = item['humans']
    params = {key: torch.tensor(humans[key]).to(device)
              for key in ['betas', 'global_orient', 'body_pose', 'scale', 'transl']}
    vertices = body_model(**params).vertices.detach()

    cam = item['cam']
    img_width, img_height = image_size
    for name in ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz', 'fl']:
        value = torch.tensor(np.array(cam[name]).reshape(1, 1)).float().to(device)
        getattr(camera, name).data[:] = value
    camera.iw[:] = float(img_width)
    camera.ih[:] = float(img_height)

    bboxes = []
    for i in range(vertices.shape[0]):
        with torch.no_grad():
            projected = camera.project(vertices[[i]])[0].cpu().numpy()
        bboxes.append([projected[:, 0].min(), projected[:, 1].min(),
                       projected[:, 0].max(), projected[:, 1].max()])

    return vertices, np.array(bboxes)


def match_result(pred_bboxes, gt_bboxes):
    """Best (score, needs_swap) between a predicted pair and the annotated pair.

    Also settles identity: the annotation's region_id pairs are ordered
    (person_ids[0], person_ids[1]), so if our human 0 turns out to be the
    annotation's second person the contact map has to be transposed. Returning
    the swap flag keeps that decision explicit rather than assumed.
    """
    straight = 0.5 * (bbox_iou(pred_bboxes[0], gt_bboxes[0])
                      + bbox_iou(pred_bboxes[1], gt_bboxes[1]))
    swapped = 0.5 * (bbox_iou(pred_bboxes[0], gt_bboxes[1])
                     + bbox_iou(pred_bboxes[1], gt_bboxes[0]))
    if swapped > straight:
        return swapped, True
    return straight, False


def gt_contact_map(anno, device):
    cmap = torch.zeros((1, NUM_REGIONS, NUM_REGIONS), dtype=torch.bool, device=device)
    for r0, r1 in anno['smplx']['region_id']:
        cmap[0, r0, r1] = True
    return cmap


def evaluate(cfg, cmd_args, runs, imgnames, annotations, image_sizes):
    device = cfg.device

    body_model = build_bodymodel(cfg=cfg.body_model, batch_size=2, device=device)
    camera = build_camera(camera_cfg=cfg.camera, camera_type=cfg.camera.type,
                          batch_size=1, device=device).to(device)
    cmapper = ContactMap(region_to_vertex=REGION_TO_VERTEX_PATH).to(device)
    contact_iou = ContactIOU(name='ContactIOU').to(device)
    pcc_x = torch.from_numpy(np.arange(0.0, 1.0, 0.05)).to(device)

    penetration_crit = None
    if not cmd_args.skip_penetration:
        penetration_crit = MaxIntersection(
            model_type='smplx',
            body_model_utils_folder='essentials/body_model_utils').to(device)

    per_image = {name: {} for name in runs}
    problems = {name: {'missing': [], 'unmatched': []} for name in runs}

    for imgname in tqdm(imgnames, desc='images'):
        record_anno = annotations[imgname]
        anno = record_anno['ci_sign'][0]
        gt_cmap = gt_contact_map(anno, device)
        if not gt_cmap.any():
            continue  # annotation lists no region pair; nothing to score against

        gt_bboxes = np.array([record_anno['bbxes'][pid]
                              for pid in anno['person_ids']], dtype=float)
        image_size = image_sizes[imgname]

        for name, folder in runs.items():
            candidates = candidate_results(folder, imgname)
            if not candidates:
                problems[name]['missing'].append(imgname)
                continue

            # Pick the predicted pair that actually corresponds to the annotated
            # pair, and learn whether our human 0 is their person 0.
            best = None
            for path in candidates:
                vertices, pred_bboxes = load_result(
                    path, body_model, camera, image_size, device)
                score, needs_swap = match_result(pred_bboxes, gt_bboxes)
                if best is None or score > best[0]:
                    best = (score, needs_swap, vertices, path)

            score, needs_swap, vertices, path = best
            if score < cmd_args.min_bbox_iou:
                problems[name]['unmatched'].append([imgname, round(score, 3)])
                continue

            order = [1, 0] if needs_swap else [0, 1]
            v0, v1 = vertices[[order[0]]], vertices[[order[1]]]

            heat = cmapper.get_full_heatmap(v0, v1)
            binary = heat < cmd_args.contact_threshold
            iou, precision, recall, fscore = contact_iou(binary, gt_cmap)

            distances = heat[gt_cmap]
            pcc = (distances[None] < pcc_x[:, None]).float().mean(1)

            record = {
                'iou': iou.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'fscore': fscore.item(),
                'dist_on_gt_mm': distances.mean().item() * 1000.0,
                'min_dist_on_gt_mm': distances.min().item() * 1000.0,
                'gt_regions': int(gt_cmap.sum().item()),
                'est_regions': int(binary.sum().item()),
                'bbox_match_iou': float(score),
                'n_candidates': len(candidates),
                'swapped': bool(needs_swap),
                'result_path': path,
                'pcc': pcc.cpu().numpy(),
            }

            if penetration_crit is not None:
                scores = score_batch(penetration_crit, v0, v1,
                                     cmd_args.penetration_lowres)
                record['penetration_mm'] = 1000.0 * max(
                    scores['max_v1_in_v2'][0], scores['max_v2_in_v1'][0])

            per_image[name][imgname] = record

    return per_image, problems, pcc_x.cpu().numpy()


def aggregate(per_image, imgnames, pcc_x):
    """Mean of every scalar metric over `imgnames`, plus the PCC curve."""
    summary = {}
    scalar_keys = ['iou', 'precision', 'recall', 'fscore', 'dist_on_gt_mm',
                   'min_dist_on_gt_mm', 'gt_regions', 'est_regions', 'penetration_mm',
                   'bbox_match_iou', 'n_candidates']
    for name, records in per_image.items():
        rows = [records[img] for img in imgnames if img in records]
        entry = {'n_images': len(rows)}
        if not rows:
            summary[name] = entry
            continue
        for key in scalar_keys:
            values = [r[key] for r in rows if key in r]
            if values:
                entry[key] = float(np.mean(values))
        pcc = np.stack([r['pcc'] for r in rows]).mean(axis=0)
        entry['pcc'] = pcc.tolist()
        for x, y in zip(pcc_x, pcc):
            entry[f'pcc@{x:.2f}'] = float(y)
        summary[name] = entry
    return summary


def write_report(cmd_args, runs, summary_own, summary_common, problems,
                 common_images, all_images, pcc_x, per_image):
    out = cmd_args.output_folder
    os.makedirs(out, exist_ok=True)

    # --- metrics.csv: one row per run, on the common subset ---
    fieldnames = ['run', 'scope', 'n_images']
    for name in summary_common:
        for key in summary_common[name]:
            if key not in fieldnames and key != 'pcc':
                fieldnames.append(key)

    with open(osp.join(out, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for scope, summary in [('common', summary_common), ('own', summary_own)]:
            for name in runs:
                writer.writerow({'run': name, 'scope': scope, **summary[name]})

    # --- per_image.csv: for eyeballing the biggest disagreements ---
    with open(osp.join(out, 'per_image.csv'), 'w', newline='') as f:
        cols = ['imgname'] + [f'{name}_{metric}' for name in runs
                              for metric in ['iou', 'fscore', 'dist_on_gt_mm',
                                             'penetration_mm']]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for imgname in sorted(common_images):
            row = {'imgname': imgname}
            for name in runs:
                record = per_image[name][imgname]
                for metric in ['iou', 'fscore', 'dist_on_gt_mm', 'penetration_mm']:
                    if metric in record:
                        row[f'{name}_{metric}'] = record[metric]
            writer.writerow(row)

    # --- pcc.png ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in runs:
        if 'pcc' in summary_common[name]:
            ax.plot(pcc_x, summary_common[name]['pcc'], '-o', markersize=3, label=name)
    ax.set_xlabel('distance threshold (m)')
    ax.set_ylabel('PCC')
    ax.set_title(f'PCC on {len(common_images)} common images')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out, 'pcc.png'), dpi=120)
    plt.close(fig)

    # --- json + markdown ---
    with open(osp.join(out, 'metrics.json'), 'w') as f:
        json.dump({
            'runs': runs,
            'n_images_considered': len(all_images),
            'n_images_common': len(common_images),
            'contact_threshold': cmd_args.contact_threshold,
            'common': summary_common,
            'own': summary_own,
            'problems': {name: {k: len(v) for k, v in p.items()}
                         for name, p in problems.items()},
            'problem_images': problems,
        }, f, indent=2)

    lines = [
        '# FlickrCI3D contact-only comparison',
        '',
        f'- images considered: {len(all_images)}',
        f'- images with a result from every run (used below): {len(common_images)}',
        f'- binary contact threshold: {cmd_args.contact_threshold} m',
        '',
        '## Coverage',
        '',
        '| run | results | no result | bbox mismatch |',
        '| --- | --- | --- | --- |',
    ]
    for name in runs:
        lines.append(f"| {name} | {summary_own[name]['n_images']} | "
                     f"{len(problems[name]['missing'])} | "
                     f"{len(problems[name]['unmatched'])} |")

    metric_cols = ['iou', 'fscore', 'precision', 'recall', 'dist_on_gt_mm',
                   'penetration_mm', 'pcc@0.10']
    metric_cols = [m for m in metric_cols if m in summary_common[list(runs)[0]]]
    lines += ['', '## Metrics on the common subset', '',
              '| run | ' + ' | '.join(metric_cols) + ' |',
              '| --- | ' + ' | '.join(['---'] * len(metric_cols)) + ' |']
    for name in runs:
        values = [f'{summary_common[name][m]:.4f}' for m in metric_cols]
        lines.append(f'| {name} | ' + ' | '.join(values) + ' |')

    with open(osp.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    guru.info('\n'.join(lines))


def main(cfg, cmd_args):
    runs = parse_runs(cmd_args.runs)
    annotations = json.load(open(cmd_args.annotations, 'r'))

    if cmd_args.subset and cmd_args.subset.lower() != 'none':
        imgnames = json.load(open(cmd_args.subset, 'r'))['imgnames']
    else:
        imgnames = sorted(k for k, v in annotations.items() if len(v['ci_sign']) == 1)

    imgnames = [x for x in imgnames if x in annotations
                and len(annotations[x]['ci_sign']) == 1]
    guru.info(f'Evaluating {len(runs)} run(s) on {len(imgnames)} images')

    image_sizes = {}
    for imgname in imgnames:
        with Image.open(osp.join(cmd_args.images_folder, f'{imgname}.png')) as img:
            image_sizes[imgname] = img.size  # (width, height)

    per_image, problems, pcc_x = evaluate(
        cfg, cmd_args, runs, imgnames, annotations, image_sizes)

    scored = [set(per_image[name].keys()) for name in runs]
    common_images = sorted(set.intersection(*scored)) if scored else []
    guru.info(f'{len(common_images)} images have a result from every run')

    summary_own = aggregate(per_image, imgnames, pcc_x)
    summary_common = aggregate(per_image, common_images, pcc_x)

    write_report(cmd_args, runs, summary_own, summary_common, problems,
                 common_images, imgnames, pcc_x, per_image)


if __name__ == '__main__':
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
