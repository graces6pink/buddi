import os
import os.path as osp
import argparse
import math
import pickle
import textwrap

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as SciRot

# the very function bev/main.py::single_image_forward calls on its input, so
# what we show is byte-for-byte what the network sees. (Note: llib has its own
# copy at llib/models/regressors/bev/utils.py, but that one references torch
# without importing it and raises NameError -- it is only reachable when
# image_processing.load_image is on, which no config here turns on.)
from romp.utils import img_preprocess, padding_image
import llib.data.preprocess.utils.process_interx_bev as PB

'''
End-to-end inspection of a SINGLE InterX frame: every intermediate of the
stage-2 pipeline, as an image grid plus a readable parameter dump.

    raw sample (stage 1)
      -> two-person SMPL-X mesh in the mocap world frame
      -> randomized camera -> rendered 512x512 "photo"
      -> the exact array handed to BEV (BGR uint8)
      -> what BEV internally makes of it (img_preprocess)
      -> BEV's raw detections
      -> detection <-> person 0/1 matching
      -> SMPL -> SMPL-X conversion
      -> the bev_smplx_* / pgt_smplx_*_cam finally written back to the sample

Everything is produced by calling the production methods on
InterXBevProcessor rather than reimplementing them, so what gets verified is
the pipeline itself and not a parallel copy of it.

Two camera modes:
  --replay (default when the sample has render_cam_*): re-render with the exact
    camera a previous process_interx_bev.py run used for this frame, and
    additionally diff the recomputed pgt_*_cam against the stored ones.
  --new-camera: draw a fresh camera. Use for frames that have not been
    processed yet, or to see the same frame from another angle.

Usage:
  python llib/data/preprocess/utils/inspect_interx_frame.py --random
  python llib/data/preprocess/utils/inspect_interx_frame.py --imgname G001T000A000R000_0_289
  python llib/data/preprocess/utils/inspect_interx_frame.py --take-id G001T000A000R004 --frame-index 2
'''

PANEL = PB.IMAGE_SIZE       # every panel is rendered/resized to this
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# frame selection
# --------------------------------------------------------------------------

def select_sample(processed, args):
    """Return (take_id, frame_index, sample)."""
    if args.imgname is not None:
        for take_id, samples in processed.items():
            for i, s in enumerate(samples):
                if s['imgname'] == args.imgname:
                    return take_id, i, s
        raise SystemExit(f'imgname {args.imgname!r} not found in processed.pkl')

    if args.take_id is not None:
        if args.take_id not in processed:
            raise SystemExit(f'take {args.take_id!r} not found in processed.pkl')
        samples = processed[args.take_id]
        if not 0 <= args.frame_index < len(samples):
            raise SystemExit(
                f'take {args.take_id} has {len(samples)} frames, '
                f'--frame-index {args.frame_index} is out of range')
        return args.take_id, args.frame_index, samples[args.frame_index]

    # --random: prefer a frame that has already been through stage 2, otherwise
    # any frame at all (the script then falls back to --new-camera).
    rng = np.random.RandomState(args.pick_seed)
    take_ids = sorted(processed.keys())
    with_cam = [t for t in take_ids if processed[t] and 'render_cam_R' in processed[t][0]]
    pool = with_cam if with_cam else take_ids
    take_id = pool[rng.randint(len(pool))]
    frame_index = rng.randint(len(processed[take_id]))
    return take_id, frame_index, processed[take_id][frame_index]


# --------------------------------------------------------------------------
# small rendering helpers
# --------------------------------------------------------------------------

def render_meshes(proc, verts_list, colors, pitch, yaw, roll):
    """Render a list of [V,3] meshes with one shared camera / center / zoom."""
    verts = torch.cat([v.reshape(1, -1, 3) for v in verts_list], dim=0)
    center = verts.reshape(-1, 3).mean(0)
    verts_c = verts - center
    radius = verts_c.reshape(-1, 3).norm(dim=-1).max().item()
    dist = radius / math.sin((PB.BEV_FOV / 2) * math.pi / 180) * 1.3
    proc.renderer.update_camera_pose(pitch, yaw, roll, 0.0, 0.0, dist)
    img_t = proc.renderer.render(verts_c, proc.body_model.faces_tensor, colors=colors)
    return (img_t[0].detach().cpu().numpy() * 255).astype(np.uint8)[..., :3]


def smplx_verts(proc, global_orient, body_pose, betas, transl):
    out = proc.body_model(
        global_orient=torch.as_tensor(np.asarray(global_orient)).float().to(proc.device),
        body_pose=torch.as_tensor(np.asarray(body_pose)).float().to(proc.device),
        betas=torch.as_tensor(np.asarray(betas)).float().to(proc.device),
        transl=torch.as_tensor(np.asarray(transl)).float().to(proc.device),
    )
    return out.vertices  # [2,V,3]


def fit_panel(img):
    """Letterbox to PANEL x PANEL. Deliberately not a plain resize: BEV's own
    rendered_image is a wide mesh|bird-view strip, and squashing it to a square
    distorts the bodies into something that looks like a bug but is not."""
    h, w = img.shape[:2]
    if (h, w) == (PANEL, PANEL):
        return img.copy()
    scale = min(PANEL / h, PANEL / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((PANEL, PANEL, 3), 255, dtype=np.uint8)
    top, left = (PANEL - nh) // 2, (PANEL - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def label_panel(img, title, subtitle=''):
    """Draw a titled frame around a panel. Input/returns RGB uint8."""
    out = fit_panel(img)
    cv2.rectangle(out, (0, 0), (PANEL - 1, PANEL - 1), (200, 200, 200), 2)
    cv2.rectangle(out, (0, 0), (PANEL - 1, 62), (30, 30, 30), -1)
    cv2.putText(out, title, (10, 26), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (10, 50), FONT, 0.45, (170, 220, 255), 1, cv2.LINE_AA)
    return out


def blank_panel(message):
    img = np.full((PANEL, PANEL, 3), 245, dtype=np.uint8)
    for i, line in enumerate(textwrap.wrap(message, 34)):
        cv2.putText(img, line, (20, 150 + 30 * i), FONT, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def grid(panels, cols=4):
    rows = []
    for r in range(0, len(panels), cols):
        row = panels[r:r + cols]
        while len(row) < cols:
            row.append(np.full((PANEL, PANEL, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------
# dump formatting
# --------------------------------------------------------------------------

class Dump:
    def __init__(self):
        self.lines = []

    def head(self, text):
        self.lines += ['', '=' * 78, text, '=' * 78]

    def sub(self, text):
        self.lines += ['', f'--- {text} ---']

    def kv(self, key, value):
        self.lines.append(f'  {key:<34s} {value}')

    def arr(self, name, a, per_row=None):
        a = np.asarray(a)
        self.lines.append(f'  {name}  shape={tuple(a.shape)} dtype={a.dtype}')
        with np.printoptions(precision=4, suppress=True, linewidth=160):
            body = str(a if per_row is None else a[:per_row])
        for line in body.splitlines():
            self.lines.append('      ' + line)

    def text(self, s=''):
        self.lines.append(s)

    def write(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines) + '\n')


def angle_between(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Inspect every intermediate of the InterX -> BEV pipeline for one frame.')
    ap.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    sel = ap.add_argument_group('frame selection (pick one)')
    sel.add_argument('--imgname', default=None, help='exact imgname, e.g. G001T000A000R000_0_289')
    sel.add_argument('--take-id', default=None)
    sel.add_argument('--frame-index', type=int, default=0, help='index within the take, with --take-id')
    sel.add_argument('--random', action='store_true', help='pick a random frame')
    sel.add_argument('--pick-seed', type=int, default=0, help='seed for --random')
    cam = ap.add_argument_group('camera')
    cam.add_argument('--new-camera', action='store_true',
                     help='draw a fresh camera instead of replaying the stored one')
    cam.add_argument('--seed', type=int, default=0, help='seed for the fresh camera')
    ap.add_argument('--out-dir', default='datasets/processed/InterX/diagnostics/frame_inspect')
    ap.add_argument('--save-panels', action='store_true', help='also save each panel separately')
    args = ap.parse_args()

    if not (args.imgname or args.take_id or args.random):
        ap.error('pick a frame with --imgname, --take-id or --random')

    os.makedirs(args.out_dir, exist_ok=True)
    processed_fn = osp.join(args.processed_data_folder, 'processed.pkl')
    print(f'loading {processed_fn} ...')
    processed = pickle.load(open(processed_fn, 'rb'))

    take_id, frame_index, sample = select_sample(processed, args)
    imgname = sample['imgname']
    print(f'inspecting {imgname}  (take {take_id}, frame index {frame_index} in take)')

    has_stored_cam = 'render_cam_R' in sample
    replay = has_stored_cam and not args.new_camera
    if args.new_camera:
        mode_note = 'fresh camera (--new-camera)'
    elif has_stored_cam:
        mode_note = 'replaying the stored camera'
    else:
        mode_note = 'fresh camera (sample has no render_cam_*, stage 2 has not reached it yet)'
        print('  NOTE: this frame has no stored camera yet -> falling back to a fresh one.')

    d = Dump()
    panels, panel_names = [], []

    # ---------------- STEP 0: the stage-1 sample ----------------
    d.head(f'STEP 0  stage-1 sample   {imgname}')
    parts = imgname.rsplit('_', 2)
    d.kv('take_id', take_id)
    if len(parts) == 3:
        d.kv('decoded imgname', f'take={parts[0]}  segment={parts[1]}  source frame={parts[2]}')
    d.kv('index within take', f'{frame_index} of {len(processed[take_id])}')
    d.kv('information_missing (stored)', sample.get('information_missing'))
    d.kv('camera mode', mode_note)
    d.sub('all keys on the stored sample')
    for k in sorted(sample.keys()):
        v = sample[k]
        if isinstance(v, np.ndarray):
            d.kv(k, f'ndarray shape={tuple(v.shape)} dtype={v.dtype}')
        else:
            d.kv(k, f'{type(v).__name__} = {v!r}')

    # ---------------- STEP 1: world-frame ground truth ----------------
    d.head('STEP 1  mocap world-frame ground truth (what stage 1 sampled)')
    for i in range(2):
        d.sub(f'person {i}')
        d.arr('pgt_smplx_global_orient (axis-angle)', sample['pgt_smplx_global_orient'][i])
        d.arr('pgt_smplx_transl', sample['pgt_smplx_transl'][i])
        d.arr('pgt_smplx_betas', sample['pgt_smplx_betas'][i])
        d.arr('pgt_smplx_scale', sample['pgt_smplx_scale'][i])
        bp = sample['pgt_smplx_body_pose'][i].reshape(21, 3)
        d.arr('pgt_smplx_body_pose (21x3, first 5 joints)', bp, per_row=5)
        d.kv('body_pose per-joint angle deg (min/med/max)',
             '{:.1f} / {:.1f} / {:.1f}'.format(
                 *np.percentile(np.degrees(np.linalg.norm(bp, axis=1)), [0, 50, 100])))
    root_dist = float(np.linalg.norm(
        sample['pgt_smplx_transl'][1] - sample['pgt_smplx_transl'][0]))
    d.sub('two-person geometry')
    d.kv('||transl_1 - transl_0||', f'{root_dist:.4f} m   (stage-1 kept frames < dist_thresh=0.6)')

    print('building body model + renderer + BEV (this loads BEV, ~20s) ...')
    proc = PB.InterXBevProcessor(seed=args.seed)

    verts_world = smplx_verts(
        proc, sample['pgt_smplx_global_orient'], sample['pgt_smplx_body_pose'],
        sample['pgt_smplx_betas'], sample['pgt_smplx_transl'])

    # world frame is +Y up, so roll=0 renders upright here
    panels.append(label_panel(
        render_meshes(proc, [verts_world[0], verts_world[1]], ['blue', 'red'], 10.0, 0.0, 0.0),
        '1. world-frame GT (front)', 'blue = person 0, red = person 1'))
    panel_names.append('01_world_front')
    panels.append(label_panel(
        render_meshes(proc, [verts_world[0], verts_world[1]], ['blue', 'red'], 10.0, 90.0, 0.0),
        '2. world-frame GT (side)', f'root distance {root_dist:.3f} m'))
    panel_names.append('02_world_side')

    # ---------------- STEP 2: the camera ----------------
    cam_override = None
    if replay:
        euler = sample['render_cam_euler']
        cam_override = {'pitch': euler[0], 'yaw': euler[1], 'roll': euler[2],
                        'dist': sample['render_cam_dist']}

    bgr_image, our_screen_xy, cam = proc.render_sample(sample, cam_override=cam_override)

    R = cam['R'].cpu().numpy()
    T = cam['T'].cpu().numpy()
    center = cam['center'].cpu().numpy()
    A = PB.BEV_FRAME_FIX @ R.T
    b = PB.BEV_FRAME_FIX @ (T - center @ R)

    d.head('STEP 2  randomized camera (world -> BEV frame)')
    d.kv('mode', mode_note)
    d.kv('pitch / yaw / roll (deg)',
         f"{cam['pitch']:.2f} / {cam['yaw']:.2f} / {cam['roll']:.2f}")
    d.kv('mesh radius about the pair centroid', f"{cam['radius']:.4f} m")
    d.kv('base_dist = radius/sin(fov/2)*1.3', f"{cam['base_dist']:.4f} m")
    d.kv('dist actually used', f"{cam['dist']:.4f} m  (jitter x{cam['dist']/cam['base_dist']:.3f})")
    d.arr('center (pair centroid, subtracted before rendering)', center)
    d.arr('R  (pytorch3d, row-vector: x_cam = x_world @ R + T)', R)
    d.arr('T', T)
    d.text()
    d.text('  column-vector form used for the SMPL parameters, x_bev = A x_world + b,')
    d.text('  with A = BEV_FRAME_FIX @ R^T   and   b = BEV_FRAME_FIX @ (T - center @ R):')
    d.arr('BEV_FRAME_FIX (pytorch3d cam -> BEV cam, 180 deg about Z)', PB.BEV_FRAME_FIX)
    d.arr('A', A)
    d.arr('b', b)
    if replay:
        d.kv('replayed euler from sample', np.round(sample['render_cam_euler'], 4).tolist())
        d.kv('max |R_recomputed - R_stored|',
             f"{np.abs(R - sample['render_cam_R']).max():.3e}")
        d.kv('max |T_recomputed - T_stored|',
             f"{np.abs(T - sample['render_cam_T']).max():.3e}")

    rgb_image = np.ascontiguousarray(bgr_image[..., ::-1])
    panels.append(label_panel(
        rgb_image, '3. rendered photo (RGB)',
        f"yaw {cam['yaw']:.0f} pitch {cam['pitch']:.0f} dist {cam['dist']:.2f}m"))
    panel_names.append('03_rendered_rgb')

    # Panel 4 shows the BGR buffer as-is. Worth knowing: the meshes are
    # rendered with colors=['gray'], so R==G==B and swapping the channel order
    # is a no-op on the pixels -- a channel-order mistake in this pipeline
    # would be both invisible here and harmless. The real check is the pixel
    # equality assertion behind panel 5.
    chan_spread = int(np.abs(rgb_image[..., 0].astype(int) - rgb_image[..., 2].astype(int)).max())
    panels.append(label_panel(
        bgr_image, '4. array handed to BEV (BGR)',
        f'max|R-B| = {chan_spread} (0 = grayscale, swap is a no-op)'))
    panel_names.append('04_bev_input_bgr')

    # ---------------- STEP 3: what BEV does to the input ----------------
    # padding_image separately only so we can report the padded shape; BEV
    # itself calls img_preprocess, which does the same padding internally.
    pad_image, _ = padding_image(cv2.cvtColor(bgr_image.copy(), cv2.COLOR_BGR2RGB))
    bev_input_t, image_pad_info = img_preprocess(bgr_image.copy(), input_size=512)
    bev_internal_rgb = bev_input_t[0].numpy().astype(np.uint8)
    pixels_match = bool(np.array_equal(bev_internal_rgb, rgb_image))

    d.head('STEP 3  the array BEV actually receives')
    d.kv('renderer output (float RGB)', 'range [0,1], converted with *255 -> uint8')
    d.kv('rgb_image', f'shape={rgb_image.shape} dtype={rgb_image.dtype} '
                      f'min={rgb_image.min()} max={rgb_image.max()}')
    d.kv('bgr_image passed to BEV', f'shape={bgr_image.shape} dtype={bgr_image.dtype} '
                                    f'contiguous={bgr_image.flags["C_CONTIGUOUS"]}')
    d.sub('BEV internal img_preprocess (romp.utils, the one bev/main.py calls)')
    d.kv('image_pad_info [top,bottom,left,right,h,w]',
         [float(x) for x in image_pad_info.tolist()])
    d.kv('padded image shape', pad_image.shape)
    d.kv('padding is a no-op', pad_image.shape[:2] == bgr_image.shape[:2])
    d.kv('resize to 512 is a no-op', pad_image.shape[0] == 512 and pad_image.shape[1] == 512)
    d.kv('tensor handed to the network', f'shape={tuple(bev_input_t.shape)} '
                                         f'dtype={bev_input_t.dtype} '
                                         f'range=[{bev_input_t.min():.1f},{bev_input_t.max():.1f}]')
    d.kv('max |R-B| over the rendered image',
         f'{int(np.abs(rgb_image[...,0].astype(int) - rgb_image[...,2].astype(int)).max())}'
         '   (0 = grayscale render, so the BGR/RGB swap cannot change any pixel)')
    d.kv('BEV internal RGB == our rendered RGB', f'{pixels_match}  <-- BGR round-trip check')
    if not pixels_match:
        diff = np.abs(bev_internal_rgb.astype(int) - rgb_image.astype(int))
        d.kv('  max abs pixel diff', diff.max())
    print(f'  BGR round-trip check (panel 3 == panel 5): {pixels_match}')

    panels.append(label_panel(
        bev_internal_rgb, '5. BEV internal (after img_preprocess)',
        f'identical to panel 3: {pixels_match}'))
    panel_names.append('05_bev_internal_rgb')

    # ---------------- STEP 4: BEV output ----------------
    bev_out = proc.bev_model(bgr_image)

    d.head('STEP 4  BEV raw output')
    if bev_out is None:
        n_det = 0
        d.kv('BEV returned', 'None (no detection at all)')
    else:
        n_det = int(bev_out['cam_trans'].shape[0])
        d.sub('all output keys')
        for k in sorted(bev_out.keys()):
            v = bev_out[k]
            d.kv(k, f'shape={tuple(v.shape)} dtype={v.dtype}'
                    if isinstance(v, np.ndarray) else type(v).__name__)
        d.sub('detections')
        d.kv('number of detections', f'{n_det}   (pipeline requires exactly 2)')
        if 'center_confs' in bev_out:
            d.arr('center_confs', bev_out['center_confs'])
        for j in range(n_det):
            d.sub(f'detection {j}')
            d.arr('smpl_thetas[:3]  (global orient, axis-angle)', bev_out['smpl_thetas'][j][:3])
            d.arr('smpl_betas (10 betas + kid scale)', bev_out['smpl_betas'][j])
            d.arr('cam (BEV weak-perspective)', bev_out['cam'][j])
            d.arr('cam_trans', bev_out['cam_trans'][j])
            pj = bev_out['pj2d_org'][j]
            d.kv('pj2d_org x range', f'[{pj[:,0].min():.1f}, {pj[:,0].max():.1f}]')
            d.kv('pj2d_org y range', f'[{pj[:,1].min():.1f}, {pj[:,1].max():.1f}]')

    ok = bev_out is not None and n_det == 2
    print(f'  BEV detections: {n_det} -> {"ok" if ok else "FRAME WOULD BE MARKED information_missing"}')

    # panel 6: keypoints over the input
    kp_img = rgb_image.copy()
    det_colors = [(0, 170, 255), (255, 90, 0), (0, 220, 120), (200, 0, 200)]
    if bev_out is not None:
        for j in range(n_det):
            col = det_colors[j % len(det_colors)]
            for (x, y) in bev_out['pj2d_org'][j][:, :2]:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(kp_img, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
            cx = float(bev_out['pj2d_org'][j][:, 0].mean())
            cv2.line(kp_img, (int(cx), 66), (int(cx), 96), col, 2)
            conf = bev_out['center_confs'][j] if 'center_confs' in bev_out else float('nan')
            # stagger the rows: two detections of a closely interacting pair
            # have nearly the same screen x, so one row would overlap
            cv2.putText(kp_img, f'det{j} conf {conf:.2f}',
                        (int(np.clip(cx - 55, 4, PANEL - 130)), 112 + 22 * j),
                        FONT, 0.45, col, 1, cv2.LINE_AA)
    for i, (sx, sy) in enumerate(our_screen_xy):
        cv2.drawMarker(kp_img, (int(sx), int(sy)), (0, 0, 0), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(kp_img, f'p{i}', (int(max(4, sx - 8)), int(max(16, sy - 16))),
                    FONT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    panels.append(label_panel(
        kp_img, '6. BEV pj2d_org + matching',
        f'{n_det} detection(s); black crosses = our screen xy'))
    panel_names.append('06_bev_keypoints')

    # panel 7: BEV's own render, if it produced one
    if bev_out is not None and 'rendered_image' in bev_out:
        rend = bev_out['rendered_image']
        rend = rend[..., :3][..., ::-1] if rend.shape[-1] >= 3 else rend
        panels.append(label_panel(
            np.ascontiguousarray(rend), '7. BEV rendered_image',
            f"show_items={getattr(proc.bev_model.settings, 'show_items', '?')}, letterboxed"))
    else:
        panels.append(label_panel(blank_panel('BEV produced no rendered_image'),
                                  '7. BEV rendered_image', 'not available'))
    panel_names.append('07_bev_rendered')

    # ---------------- STEP 5 + 6: matching, conversion, alignment ----------------
    gt_cam = proc.pgt_to_camera_frame(sample, cam)

    d.head('STEP 5  detection matching and SMPL -> SMPL-X conversion')
    if not ok:
        d.kv('SKIPPED', 'BEV did not return exactly 2 detections; '
                        'process_sample() would set information_missing=True here')
        bev_fields = None
    else:
        bev_xy = bev_out['pj2d_org'][:, :, :2].mean(axis=1)
        cost = np.linalg.norm(our_screen_xy[:, None, :] - bev_xy[None, :, :], axis=-1)
        pairing = proc.match_detections(our_screen_xy, bev_out['pj2d_org'])
        d.sub('matching (screen-space 2D nearest neighbour)')
        d.kv('separation (px)', f'{proc.separation(our_screen_xy):.1f} '
                                f'(guard rejects below {PB.MIN_SEPARATION_PX:.0f})')
        d.arr('our_screen_xy (person 0, 1)', our_screen_xy)
        d.arr('bev_screen_xy (det 0, 1)', bev_xy)
        d.arr('cost matrix ||our_i - bev_j||', cost)
        d.kv('cost identity  (0->0, 1->1)', f'{cost[0,0] + cost[1,1]:.2f}')
        d.kv('cost swapped   (0->1, 1->0)', f'{cost[0,1] + cost[1,0]:.2f}')
        d.kv('chosen pairing [det for p0, det for p1]', pairing)

        d.sub('SMPL -> SMPL-X per person (mirrors hi4d.py::process_bev)')
        for i, det_idx in enumerate(pairing):
            bs = bev_out['smpl_betas'][det_idx]
            smpl_scale = bs[-1]
            d.kv(f'person {i} <- detection {det_idx}', '')
            d.kv('  smpl kid scale', f'{smpl_scale:.4f}  -> '
                                     f'{"SMIL (child)" if smpl_scale > 0.8 else "SMPLA (adult)"} converter')
            d.arr('  input smpl_betas (10+scale)', bs)

        bev_fields = proc.bev_to_smplx(bev_out, pairing)
        for i in range(2):
            d.sub(f'converted person {i}')
            d.arr('bev_smplx_betas (10)', bev_fields['bev_smplx_betas'][i])
            d.arr('bev_smplx_scale', bev_fields['bev_smplx_scale'][i])
            d.arr('bev_smplx_global_orient', bev_fields['bev_smplx_global_orient'][i])
            d.arr('bev_smplx_transl', bev_fields['bev_smplx_transl'][i])

    d.head('STEP 6  what gets written back, and frame alignment')
    d.sub('pgt_smplx_*_cam (ground truth moved into BEV\'s frame)')
    for i in range(2):
        d.arr(f'person {i} pgt_smplx_global_orient_cam', gt_cam['pgt_smplx_global_orient_cam'][i])
        d.arr(f'person {i} pgt_smplx_transl_cam', gt_cam['pgt_smplx_transl_cam'][i])

    if replay:
        do = np.abs(gt_cam['pgt_smplx_global_orient_cam'] - sample['pgt_smplx_global_orient_cam']).max()
        dt = np.abs(gt_cam['pgt_smplx_transl_cam'] - sample['pgt_smplx_transl_cam']).max()
        d.sub('recomputed vs stored (replay mode -- both should be ~1e-6)')
        d.kv('max |global_orient_cam recomputed - stored|', f'{do:.3e}')
        d.kv('max |transl_cam recomputed - stored|', f'{dt:.3e}')
        print(f'  replay diff vs stored: orient {do:.2e}, transl {dt:.2e}')

    if bev_fields is not None:
        d.sub('alignment: BEV estimate vs camera-frame ground truth')
        Rb = SciRot.from_rotvec(bev_fields['bev_smplx_global_orient'])
        Rg_cam = SciRot.from_rotvec(gt_cam['pgt_smplx_global_orient_cam'])
        for i in range(2):
            d.kv(f'person {i} orientation residual',
                 f'{np.degrees((Rb[i] * Rg_cam[i].inv()).magnitude()):.1f} deg')
        t_bev = bev_fields['bev_smplx_transl'][1] - bev_fields['bev_smplx_transl'][0]
        t_cam = gt_cam['pgt_smplx_transl_cam'][1] - gt_cam['pgt_smplx_transl_cam'][0]
        t_world = sample['pgt_smplx_transl'][1] - sample['pgt_smplx_transl'][0]
        d.kv('offset (t1-t0) angle, cam-frame GT vs BEV', f'{angle_between(t_cam, t_bev):.1f} deg')
        d.kv('offset (t1-t0) angle, WORLD GT vs BEV',
             f'{angle_between(t_world, t_bev):.1f} deg   <-- what it was before the fix')
        d.kv('||t1-t0||  cam GT / BEV',
             f'{np.linalg.norm(t_cam):.3f} m / {np.linalg.norm(t_bev):.3f} m')

        verts_gt_cam = smplx_verts(
            proc, gt_cam['pgt_smplx_global_orient_cam'], sample['pgt_smplx_body_pose'],
            sample['pgt_smplx_betas'], gt_cam['pgt_smplx_transl_cam'])
        verts_bev = smplx_verts(
            proc, bev_fields['bev_smplx_global_orient'], bev_fields['bev_smplx_body_pose'],
            bev_fields['bev_smplx_betas'], bev_fields['bev_smplx_transl'])
        overlay = render_meshes(
            proc,
            [verts_gt_cam[0], verts_gt_cam[1], verts_bev[0], verts_bev[1]],
            ['blue', 'red', 'green', 'yellow'],
            10.0, 30.0, 180.0)  # both in BEV's +Y-down frame -> roll 180 to view upright
        panels.append(label_panel(
            overlay, '8. GT(cam) vs BEV overlay',
            f'offset angle {angle_between(t_cam, t_bev):.0f} deg (was '
            f'{angle_between(t_world, t_bev):.0f})'))
    else:
        panels.append(label_panel(
            blank_panel('no BEV estimate for this frame, nothing to overlay'),
            '8. GT(cam) vs BEV overlay', 'skipped'))
    panel_names.append('08_alignment_overlay')

    # ---------------- outputs ----------------
    safe = imgname.replace('/', '_')
    suffix = 'replay' if replay else 'newcam'
    sheet = grid(panels, cols=4)
    sheet_fn = osp.join(args.out_dir, f'{safe}_{suffix}_pipeline.png')
    cv2.imwrite(sheet_fn, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    if args.save_panels:
        for name, panel in zip(panel_names, panels):
            cv2.imwrite(osp.join(args.out_dir, f'{safe}_{suffix}_{name}.png'),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    txt_fn = osp.join(args.out_dir, f'{safe}_{suffix}_params.txt')
    d.write(txt_fn)

    npz = {
        'imgname': imgname, 'take_id': take_id, 'frame_index': frame_index,
        'replay': replay, 'num_detections': n_det, 'pixels_match': pixels_match,
        'cam_R': R, 'cam_T': T, 'cam_center': center, 'cam_A': A, 'cam_b': b,
        'cam_euler': np.array([cam['pitch'], cam['yaw'], cam['roll']], dtype=np.float32),
        'cam_dist': np.float32(cam['dist']), 'our_screen_xy': our_screen_xy,
        'rendered_rgb': rgb_image, 'bev_input_bgr': bgr_image,
    }
    for k in ['pgt_smplx_global_orient', 'pgt_smplx_body_pose', 'pgt_smplx_betas',
              'pgt_smplx_transl', 'pgt_smplx_scale']:
        npz[k] = sample[k]
    npz.update(gt_cam)
    if bev_out is not None:
        for k, v in bev_out.items():
            if isinstance(v, np.ndarray):
                npz[f'bevraw_{k}'] = v
    if bev_fields is not None:
        npz.update(bev_fields)
    npz_fn = osp.join(args.out_dir, f'{safe}_{suffix}_params.npz')
    np.savez_compressed(npz_fn, **npz)

    print('\nwrote:')
    print(f'  {sheet_fn}')
    print(f'  {txt_fn}')
    print(f'  {npz_fn}')


if __name__ == '__main__':
    main()
