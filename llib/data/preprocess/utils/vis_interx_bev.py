"""
Render side-by-side comparisons of InterX training samples: pgt_smplx_*
(mocap ground truth, what the diffusion model is trained to predict) vs
bev_smplx_* (the BEV estimate baked into the same sample by
process_interx_bev.py, used as conditioning for the BEV-conditional model).

Person 0 = blue, person 1 = red, in both renders, so per-person pose
divergence is easy to spot at a glance. Each mesh is independently centered
and auto-zoomed (own extent -> own camera distance) since pgt lives in the
mocap world frame and bev lives in its own synthetic-camera frame -- they
are NOT directly comparable in absolute position, only in per-person pose /
two-person relative layout.
"""
import argparse
import math
import os
import os.path as osp
import pickle

import numpy as np
import torch
import smplx
import cv2

from llib.cameras.perspective import PerspectiveCamera
from llib.visualization.renderer import Pytorch3dRenderer

FOV = 60.0
IMG = 512


def build_body_model(device):
    return smplx.create(
        model_path='essentials/body_models', model_type='smplx',
        gender='neutral', num_betas=10, batch_size=2,
    ).to(device)


def make_renderer(device):
    camera = PerspectiveCamera(
        rotation=torch.tensor([[0., 0., 0.]]),
        translation=torch.tensor([[0., 0., 0.]]),
        afov_horizontal=torch.tensor([FOV]),
        image_size=torch.tensor([[IMG, IMG]]),
        batch_size=1, device=device,
    )
    renderer = Pytorch3dRenderer(cameras=camera.cameras, image_width=IMG, image_height=IMG)
    return renderer


def render_pose(renderer, body_model, global_orient, body_pose, betas, transl, device, roll=0.0):
    go = torch.from_numpy(np.asarray(global_orient)).float().to(device)
    bp = torch.from_numpy(np.asarray(body_pose)).float().to(device)
    b = torch.from_numpy(np.asarray(betas)).float().to(device)
    tr = torch.from_numpy(np.asarray(transl)).float().to(device)

    out = body_model(global_orient=go, body_pose=bp, betas=b, transl=tr)
    verts = out.vertices  # [2, V, 3]

    center = verts.reshape(-1, 3).mean(0)
    verts_c = verts - center
    radius = verts_c.reshape(-1, 3).norm(dim=-1).max().item()
    fov_half = (FOV / 2) * math.pi / 180
    dist = radius / math.sin(fov_half) * 1.3

    renderer.update_camera_pose(10.0, 30.0, roll, 0.0, 0.0, dist)
    img_t = renderer.render(verts_c, body_model.faces_tensor, colors=['blue', 'red'])
    img = (img_t[0].detach().cpu().numpy() * 255).astype(np.uint8)[..., :3]
    return img


def render_overlay(renderer, body_model, s, device, pitch=10.0, yaw=30.0):
    """Draw the camera-frame ground truth and the BEV estimate into ONE image
    with ONE camera.

    This only means anything once pgt_smplx_*_cam exists: those fields are the
    mocap ground truth expressed in BEV's own frame (see
    process_interx_bev.py::pgt_to_camera_frame), so a correct frame alignment
    shows up as the two pairs of bodies sitting on top of each other. Before
    that fix the two lived in frames differing by a per-frame random camera
    rotation and this overlay would be nonsense, which is exactly the point.

    pgt = blue / red, bev = green / yellow, person 0 first in both.
    """
    def verts_of(go, bp, be, tr):
        out = body_model(
            global_orient=torch.from_numpy(np.asarray(go)).float().to(device),
            body_pose=torch.from_numpy(np.asarray(bp)).float().to(device),
            betas=torch.from_numpy(np.asarray(be)).float().to(device),
            transl=torch.from_numpy(np.asarray(tr)).float().to(device),
        )
        return out.vertices

    v_pgt = verts_of(s['pgt_smplx_global_orient_cam'], s['pgt_smplx_body_pose'],
                     s['pgt_smplx_betas'], s['pgt_smplx_transl_cam'])
    v_bev = verts_of(s['bev_smplx_global_orient'], s['bev_smplx_body_pose'],
                     s['bev_smplx_betas'], s['bev_smplx_transl'])

    # one shared center/zoom, otherwise per-mesh centering would hide exactly
    # the misalignment we are looking for
    allv = torch.cat([v_pgt, v_bev], dim=0)
    center = allv.reshape(-1, 3).mean(0)
    allv_c = allv - center
    radius = allv_c.reshape(-1, 3).norm(dim=-1).max().item()
    dist = radius / math.sin((FOV / 2) * math.pi / 180) * 1.3

    # both meshes are in BEV's +Y-down frame, so roll it upright for the preview
    renderer.update_camera_pose(pitch, yaw, 180.0, 0.0, 0.0, dist)
    img_t = renderer.render(allv_c, body_model.faces_tensor,
                            colors=['blue', 'red', 'green', 'yellow'])
    return (img_t[0].detach().cpu().numpy() * 255).astype(np.uint8)[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/processed/InterX/train_diffusion.pkl')
    ap.add_argument('--num-samples', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='datasets/processed/InterX/diagnostics/bev_vis')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    valid = [d for d in data if not d['information_missing']]
    print(f'{len(valid)}/{len(data)} samples have a real BEV estimate (information_missing=False)')

    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(valid), size=min(args.num_samples, len(valid)), replace=False)

    body_model = build_body_model(device)
    renderer = make_renderer(device)

    has_cam = 'pgt_smplx_transl_cam' in valid[0]
    print('camera-frame ground truth present:', has_cam,
          '-- overlay panel is only drawn when it is' if not has_cam else '')

    for i, idx in enumerate(idxs):
        s = valid[idx]
        panels, labels = [], []

        if has_cam:
            # Both already in BEV's frame, so neither needs the roll=180 fix and
            # the two panels are directly comparable pose for pose.
            panels.append(render_pose(
                renderer, body_model,
                s['pgt_smplx_global_orient_cam'], s['pgt_smplx_body_pose'],
                s['pgt_smplx_betas'], s['pgt_smplx_transl_cam'], device,
                roll=180.0))
            labels.append('PGT (cam frame)')
        else:
            panels.append(render_pose(
                renderer, body_model,
                s['pgt_smplx_global_orient'], s['pgt_smplx_body_pose'],
                s['pgt_smplx_betas'], s['pgt_smplx_transl'], device))
            labels.append('PGT (world frame)')

        panels.append(render_pose(
            renderer, body_model,
            s['bev_smplx_global_orient'], s['bev_smplx_body_pose'],
            s['bev_smplx_betas'], s['bev_smplx_transl'], device,
            # BEV and the pytorch3d camera differ by 180 degrees about Z --
            # same convention as llib/data/preprocess/hi4d.py:195. With the
            # camera-frame ground truth that offset is baked into pgt_*_cam
            # too, so now both panels roll by the same amount.
            roll=180.0,
        ))
        labels.append('BEV estimate')

        if has_cam:
            panels.append(render_overlay(renderer, body_model, s, device))
            labels.append('overlay (should align)')

        gutter = 255 * np.ones((IMG, 6, 3), dtype=np.uint8)
        stacked = []
        for j, panel in enumerate(panels):
            if j:
                stacked.append(gutter)
            stacked.append(panel)
        combined = np.concatenate(stacked, axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        for j, label in enumerate(labels):
            cv2.putText(combined_bgr, label, (10 + j * (IMG + 6), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        safe_name = s['imgname'].replace('/', '_')
        out_fn = osp.join(args.out_dir, f'{safe_name}.png')
        cv2.imwrite(out_fn, combined_bgr)
        print(f'[{i+1}/{len(idxs)}] take={s.get("take_id")} imgname={s["imgname"]} -> {out_fn}')


if __name__ == '__main__':
    main()
