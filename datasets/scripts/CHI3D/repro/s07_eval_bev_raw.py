"""Evaluate raw SMPL BEV against CHI3D GT -- the paper's actual "BEV" row.

The BEV row of Table 3 (50 / 52 / 96) reports BEV's own SMPL output. Our
bev_init_chi3d.yaml run instead reports BEV *after* the SMPL->SMPL-X
ShapeConverter, because that is what the fitting pipeline loads, so it is not
strictly the same quantity.

chi3d_eval.py cannot do the raw version: it has no --predicted-is-bev flag (the
Flickr eval has one, though even there get_bev_smplx_params is dead code), and it
reads predictions from pkl files written by the optimizer. But every piece is
already sitting in the repo unused:

  * evaluation/utils.py:109  chi3d_bev_verts_from_processed_data() pulls
    bev_smpl_vertices + bev_cam_trans straight out of images_contact_processed.pkl.
    Defined, never called anywhere.
  * chi3d_eval.py:38-41      SMPL_TO_H36M and H36M_TO_J14 are loaded and then
    never used -- exactly what you need to take raw SMPL (6890 verts) to the 14
    LSP joints, since the SMPL-X J14_REGRESSOR does not apply to a SMPL mesh.

So the author evidently intended this branch and left it unfinished. This script
finishes it, mirroring chi3d_eval.py's main() metric-for-metric, and needs no
fitting run at all -- BEV is read directly from the processed pkl.

Both sides end up as 14 LSP joints in the camera frame:
    prediction   SMPL   (2, 6890,  3) -> SMPL_TO_H36M -> [H36M_TO_J14]
    ground truth SMPL-X (2, 10475, 3) -> J14_REGRESSOR
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys

import numpy as np
import smplx
import torch
from tqdm import tqdm

from llib.utils.metrics.build import build_metric
from llib.defaults.main import config as default_config, merge as merge_configs
from llib.methods.hhcs_optimization.evaluation.utils import (
    J14_REGRESSOR, SMPL_TO_H36M, H36M_TO_J14,
    chi3d_items_for_eval, chi3d_get_smplx_gt, chi3d_read_cam_params,
    chi3d_verts_world2cam, chi3d_bev_verts_from_processed_data,
    verts2joints, ResultLogger,
)

ORIG_DATA_FOLDER = 'datasets/original/CHI3D'
PROCESSED_DATA_FOLDER = 'datasets/processed/CHI3D'


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs', nargs='+',
                        default=['llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml'])
    parser.add_argument('--exp-opts', default=[], dest='exp_opts', nargs='*')
    parser.add_argument('--eval-split', default='val', choices=['train', 'val'])
    parser.add_argument('--output-folder', type=str,
                        default='demo/optimization/chi3d/eval_bev_raw',
                        help='where results.pkl goes (no predictions are read from here)')
    cmd_args = parser.parse_args()
    return merge_configs(cmd_args, default_config), cmd_args


def bev_smpl_to_j14(verts_smpl):
    """(2, 6890, 3) SMPL vertices -> list of two (1, 14, 3) LSP joint tensors."""
    j_h36m = torch.matmul(SMPL_TO_H36M, verts_smpl.float())   # (2, 17, 3)
    j14 = j_h36m[:, H36M_TO_J14, :]                            # (2, 14, 3)
    return [j14[0][None], j14[1][None]]


def main():
    cfg, cmd_args = parse_args()

    processed_data = pickle.load(
        open(f'{PROCESSED_DATA_FOLDER}/train/images_contact_processed.pkl', 'rb'))
    train_val_split = np.load(f'{PROCESSED_DATA_FOLDER}/train/train_val_split.npz')
    subjects = train_val_split[cmd_args.eval_split]
    split_folder = 'train'

    scale_mpjpe_metric = build_metric(cfg.evaluation.scale_mpjpe)
    mpjpe_metric = build_metric(cfg.evaluation.mpjpe)
    pa_mpjpe_metric = build_metric(cfg.evaluation.pa_mpjpe)
    pairwise_pa_mpjpe_metric = build_metric(cfg.evaluation.pairwise_pa_mpjpe)

    bm_smplx = smplx.create(model_path='essentials/body_models',
                            model_type='smplx', batch_size=2).to(cfg.device)

    os.makedirs(cmd_args.output_folder, exist_ok=True)
    results = ResultLogger(method_names=['est'],
                           output_fn=f'{cmd_args.output_folder}/results.pkl')
    results.info = {'actions': [], 'subjects': [], 'contact_counts': [], 'img_names': []}

    items = chi3d_items_for_eval(subjects, split_folder, ORIG_DATA_FOLDER, processed_data)

    n_eval = 0
    for subject, actions_dict in tqdm(items.items()):
        annotations = json.load(open(osp.join(
            ORIG_DATA_FOLDER, split_folder, subject, 'interaction_contact_signature.json')))
        orig_subject_folder = osp.join(ORIG_DATA_FOLDER, split_folder, subject)

        for action, cameras in actions_dict.items():
            frame_id = annotations[action]['fr_id']
            smpl_path = f'{orig_subject_folder}/smplx/{action}.json'
            _, verts_gt, _ = chi3d_get_smplx_gt(smpl_path, [frame_id], bm_smplx)
            verts_gt = torch.from_numpy(verts_gt).to(cfg.device).float()

            for cam in cameras:
                cam_params = chi3d_read_cam_params(
                    f'{orig_subject_folder}/camera_parameters/{cam}/{action}.json')
                verts_gt_camera = chi3d_verts_world2cam(verts_gt, cam_params)

                # prediction: raw SMPL from BEV, already in the BEV camera frame
                bev_verts = chi3d_bev_verts_from_processed_data(
                    processed_data, subject, action, cam, frame_id, device=cfg.device)

                est_joints = bev_smpl_to_j14(bev_verts)
                gt_joints = [verts2joints(verts_gt_camera[0][None], J14_REGRESSOR),
                             verts2joints(verts_gt_camera[1][None], J14_REGRESSOR)]

                # same person-order disambiguation as chi3d_eval.py:235-240
                onetwo = pairwise_pa_mpjpe_metric(
                    torch.cat(est_joints, dim=1).cpu().numpy(),
                    torch.cat(gt_joints, dim=1).cpu().numpy()).mean()
                twoone = pairwise_pa_mpjpe_metric(
                    torch.cat(est_joints, dim=1).cpu().numpy(),
                    torch.cat([gt_joints[1], gt_joints[0]], dim=1).cpu().numpy()).mean()
                if twoone < onetwo:
                    gt_joints = [gt_joints[1], gt_joints[0]]

                results.info['img_names'].append(f'{subject}_{action}_{frame_id:06d}_{cam}_0')
                results.info['actions'].append(action.split(' ')[0])
                results.info['subjects'].append(subject)
                results.info['contact_counts'].append(0)

                results.output['est_mpjpe_h0'].append(
                    mpjpe_metric(est_joints[0], gt_joints[0]).mean())
                results.output['est_mpjpe_h1'].append(
                    mpjpe_metric(est_joints[1], gt_joints[1]).mean())
                results.output['est_scale_mpjpe_h0'].append(scale_mpjpe_metric(
                    est_joints[0].cpu().numpy(), gt_joints[0].cpu().numpy()).mean())
                results.output['est_scale_mpjpe_h1'].append(scale_mpjpe_metric(
                    est_joints[1].cpu().numpy(), gt_joints[1].cpu().numpy()).mean())
                results.output['est_pa_mpjpe_h0'].append(pa_mpjpe_metric(
                    est_joints[0].cpu().numpy(), gt_joints[0].cpu().numpy()).mean())
                results.output['est_pa_mpjpe_h1'].append(pa_mpjpe_metric(
                    est_joints[1].cpu().numpy(), gt_joints[1].cpu().numpy()).mean())
                results.output['est_pa_mpjpe_h0h1'].append(pairwise_pa_mpjpe_metric(
                    torch.cat(est_joints, dim=1).cpu().numpy(),
                    torch.cat(gt_joints, dim=1).cpu().numpy()).mean())
                n_eval += 1

    for metric in ['est_mpjpe_h0', 'est_mpjpe_h1', 'est_pa_mpjpe_h0h1']:
        for action in sorted(set(results.info['actions'])):
            results.get_action_mean(metric, results.info['actions'], action)

    print(f'\nevaluated {n_eval} (subject, action, camera) items')
    results.topkl(print_result=True)
    print('\nTable 3 "BEV" row:  PER PERSON = est_pa_mpjpe_h0 / est_pa_mpjpe_h1')
    print('                    JOINT      = est_pa_mpjpe_h0h1')


if __name__ == '__main__':
    main()
