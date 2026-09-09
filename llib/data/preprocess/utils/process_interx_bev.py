import os
import os.path as osp
import argparse
import math
import pickle
import numpy as np
import torch
import smplx
from tqdm import tqdm
from loguru import logger as guru
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, euler_angles_to_matrix

from llib.cameras.perspective import PerspectiveCamera
from llib.visualization.renderer import Pytorch3dRenderer
from llib.data.preprocess.utils.shape_converter import ShapeConverter

'''
Stage 2 of the InterX preprocessing pipeline: adds real bev_smplx_* fields to
the samples already produced by process_interx.py (which currently zero-fills
them, since the unconditional variant never reads them). For each sample:
  1) render the two-person SMPL-X mesh from six cameras orbiting the pair, one
     every 60 degrees (see sample_camera_params), turning each mocap frame into
     six training samples the way CHI3D's four camera rigs do,
  2) run each view through BEV (loaded once, kept resident -- NOT the `bev` CLI,
     which reloads the model every call and would take ~80h at this scale),
  3) match BEV's (order-unstable) detections back to our known person 0/1 by
     screen-space position, rejecting views where the two people project too
     close together to tell apart (MIN_SEPARATION_PX),
  4) convert BEV's SMPL output to SMPL-X via ShapeConverter (same approach as
     llib/data/preprocess/hi4d.py::process_bev), and write the result in place,
  5) express the mocap ground truth in that same camera frame and store it as
     pgt_smplx_{global_orient,transl}_cam.

Step 5 is what makes the BEV fields usable as conditioning at all. The mocap
ground truth lives in the Inter-X world frame while BEV reports in the frame of
whatever camera we happened to render from, and that camera is re-drawn at
random for every single frame. Training a BEV-conditioned model on the raw
world-frame target therefore asks it to predict an orientation that the
conditioning signal genuinely does not determine. Every other dataset in this
repo already stores a camera-frame target for exactly this reason -- see
global_orient_cam in llib/data/preprocess/chi3d.py and
global_orient_cam_smplx in llib/data/preprocess/hi4d.py.

Views where BEV doesn't detect exactly 2 people, or where the projection is too
degenerate to pair them, are marked information_missing=True (existing field,
not a new one) rather than dropped, so downstream training can choose to filter
them via allow_missing_information. The camera-frame ground truth is written for
those views too -- it only depends on the camera we picked, not on BEV
succeeding.

Reads processed.pkl and writes a separate processed_mv.pkl rather than editing
in place: the stage-1 file is the source, and running this twice over its own
output would give 36 views per frame.
'''

BEV_FOV = 60.0
IMAGE_SIZE = 512

# Below this screen-space distance between the two people the projection is
# degenerate and assigning BEV's two detections to person 0/1 is a coin flip.
# Set from data (--report bins the flip rate by separation), not guessed: flips
# concentrate almost entirely under 10px -- 55.6% of those frames came out with
# the person0->person1 offset vector reversed -- are 8.7% in 10-20px and ~0%
# above. 12px is the 5th percentile of the separation distribution, so the guard
# costs about 5% of frames. Matching on 2D screen position instead of x alone
# only moves the overall flip rate 4.3% -> 3.3% and does nothing for the sub-10px
# cases: the guard is what actually fixes them. A wrong pairing is worse than a
# missing one, because information_missing does not catch it and it enters
# training as confident but false conditioning. 0 disables the guard.
#
# Re-check this against --report after the six-view change: orbiting the full 360
# adds occluded viewpoints, so the separation distribution shifts left.
MIN_SEPARATION_PX = 12.0

# Constant rotation taking a point from the PyTorch3D camera frame we render in
# (+X left, +Y up, +Z into the scene) to the frame BEV reports its SMPL
# parameters in (+X right, +Y down, same +Z) -- i.e. 180 degrees about Z. It is
# a fixed property of the two conventions, not something that varies per frame.
#
# Measured, not read off documentation: `--calibrate N` renders N random takes,
# runs BEV on them and reports how well each candidate constant lines our
# camera-frame ground truth up with BEV's estimate. On 505 frames:
#
#   candidate F                 orient p50   layout p50
#   identity                        178.0d       120.7d
#   flip_x (180 about X)            174.2d       150.4d
#   flip_y (180 about Y)            174.4d        71.8d
#   flip_z (180 about Z)  <- this    11.1d        13.7d
#
# The 11 degrees left over is BEV's own orientation error, which independently
# matches the frame-invariant relative-orientation error of the same data
# (p50 7.9d). Re-run --calibrate if the renderer or the BEV version changes.
BEV_FRAME_FIX = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


NUM_VIEWS = 6


def sample_camera_params(rng, num_views=NUM_VIEWS):
    """Cameras for one frame: a stratified orbit, one view every 360/num_views degrees.

    Mirrors what CHI3D gives the conditional model for free -- the paper uses all
    four of its camera rigs per frame ("we use all camera views of the MoCap
    datasets, i.e. 4/8 cameras for CHI3D/Hi4D"), and that multi-view coverage is
    its only viewpoint augmentation, since the mirror/rotation/scale knobs in
    llib/data/single.py are dead code and only the person swap survives.

    Only the yaw is coordinated across views: one random offset inside the first
    bin, then a fixed step. pitch/roll/distance are drawn per view, because these
    six end up as six independent training samples and sharing the framing would
    only make them more correlated.

    The full 360 is deliberate. The old sampler drew yaw ~ U(-150, 150) to avoid
    shooting from directly behind the pair, but with relative_orient=false that
    prior writes a global heading bias straight into the dataset, and CHI3D's own
    rigs surround the subjects anyway.

    Returns a list of (pitch, yaw, roll, dist_jitter).
    """
    yaw0 = rng.uniform(0, 360.0 / num_views)
    views = []
    for k in range(num_views):
        yaw = (yaw0 + k * 360.0 / num_views + 180.0) % 360.0 - 180.0
        views.append((
            float(np.clip(rng.normal(5, 12), -20, 35)),   # pitch: slightly downward, as a
                                                          # handheld shot of a standing pair
            float(yaw),
            float(np.clip(rng.normal(0, 3), -8, 8)),      # roll: handheld jitter
            float(rng.uniform(0.85, 1.25)),               # distance jitter around "just fits"
        ))
    return views


class InterXBevProcessor:

    def __init__(self, device='cuda', seed=0):
        from bev.main import BEV, bev_settings

        self.device = device
        self.rng = np.random.RandomState(seed)

        self.body_model = smplx.create(
            model_path='essentials/body_models', model_type='smplx',
            gender='neutral', num_betas=10, batch_size=2,
        ).to(device)

        camera = PerspectiveCamera(
            rotation=torch.tensor([[0., 0., 0.]]),
            translation=torch.tensor([[0., 0., 0.]]),
            afov_horizontal=torch.tensor([BEV_FOV]),
            image_size=torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]]),
            batch_size=1, device=device,
        )
        self.camera = camera
        self.renderer = Pytorch3dRenderer(
            cameras=camera.cameras, image_width=IMAGE_SIZE, image_height=IMAGE_SIZE)

        settings = bev_settings(input_args=['-i', 'dummy', '--calc_smpl'])
        self.bev_model = BEV(settings)

        # same conversion path as llib/data/preprocess/hi4d.py::process_bev
        self.shape_converter_smpla = ShapeConverter(inbm_type='smpla', outbm_type='smplxa')
        self.shape_converter_smil = ShapeConverter(inbm_type='smil', outbm_type='smplxa')
        self.bev_body_model = self.shape_converter_smpla.outbm

    def set_camera_pose(self, pitch, yaw, roll, dist):
        """Place the camera at `dist` from the scene centre, turntable style.

        Deliberately not llib/visualization/renderer.py::update_camera_pose.
        That one builds R from euler order "XYZ" -> R = Rx(pitch) Ry(yaw)
        Rz(roll), which under PyTorch3D's row-vector convention
        (x_cam = x_world @ R) applies pitch about the *world* X axis before yaw
        about world Y. Pitching about an axis that yaw has already tilted leaves
        the horizon rotated by roughly pitch*sin(yaw): level at yaw 0/180, but
        tilted by the full pitch angle at yaw +-90.

        That is not a cosmetic issue. With pitch ~ N(5,12) clipped to [-20,35]
        and yaw uniform, the effective camera roll was `roll + pitch*sin(yaw)`
        rather than the `roll ~ N(0,3) clipped to +-8` the sampler documents:
        30% of frames came out tilted more than 10 degrees, p99 28.6 degrees.
        BEV is trained on real photographs, which essentially never look like
        that, so it was a self-inflicted domain gap -- and one that correlates
        with viewing azimuth, which would make conditioning quality depend on
        which of the six views a sample came from.

        Order "YXZ" on [yaw, pitch, roll] gives R = Ry(yaw) Rx(pitch) Rz(roll),
        i.e. yaw first, so the horizon tilt is just `roll`. Framing is unchanged
        either way: the scene centre maps to (0, 0, dist) whatever R is. Writes
        cameras.R/T in place, so pgt_to_camera_frame -- which reads cameras.R --
        stays consistent automatically.
        """
        R = euler_angles_to_matrix(
            torch.tensor([[yaw, pitch, roll]], device=self.device) * math.pi / 180, "YXZ")
        self.renderer.cameras.R = R
        self.renderer.cameras.T = torch.tensor([[0.0, 0.0, dist]], device=self.device)
        self.renderer.renderer.shader.lights.location = self.renderer.cameras.get_camera_center()

    @torch.no_grad()
    def prepare_frame(self, sample):
        """Everything about a frame that does not depend on the camera.

        Split out from render_view so the six views of a frame share it. The two
        SMPL-X forwards here are the expensive part and neither involves the
        camera: the posed mesh, and the rest-pose pelvis J0 that
        pgt_to_camera_frame needs to rewrite the parameters (root rotations act
        about J0, so the change of frame cannot be applied without it). Calling
        render+convert six times instead would run twelve forwards per frame.
        """
        go = torch.from_numpy(sample['pgt_smplx_global_orient']).float().to(self.device)
        bp = torch.from_numpy(sample['pgt_smplx_body_pose']).float().to(self.device)
        betas = torch.from_numpy(sample['pgt_smplx_betas']).float().to(self.device)
        transl = torch.from_numpy(sample['pgt_smplx_transl']).float().to(self.device)

        verts = self.body_model(
            global_orient=go, body_pose=bp, betas=betas, transl=transl).vertices  # [2,V,3]
        center = verts.reshape(-1, 3).mean(0)
        verts_c = verts - center
        radius = verts_c.reshape(-1, 3).norm(dim=-1).max().item()

        return {
            'verts_c': verts_c,
            'center': center,
            'radius': radius,
            # distance at which the pair just fills the frame, with 1.3x margin
            'base_dist': radius / math.sin((BEV_FOV / 2) * math.pi / 180) * 1.3,
            'J0': self.body_model(betas=betas).joints[:, 0, :],   # [2,3] rest-pose pelvis
        }

    @torch.no_grad()
    def render_view(self, frame_ctx, cam_params):
        """Render one camera of a prepared frame into the 512x512 image BEV sees.

        cam_params is (pitch, yaw, roll, dist_jitter) from sample_camera_params,
        or (pitch, yaw, roll, dist_absolute) with jitter_is_absolute=True when
        replaying a camera stored on an existing sample -- the stored
        render_cam_dist is already multiplied out.
        """
        pitch, yaw, roll, dist = cam_params
        self.set_camera_pose(pitch, yaw, roll, dist)

        mesh = self.renderer.build_meshes(
            frame_ctx['verts_c'], self.body_model.faces_tensor, colors=['gray'])
        img_t = self.renderer.renderer(mesh)
        color_image = (img_t[0].detach().cpu().numpy() * 255).astype(np.uint8)[..., :3]
        bgr_image = np.ascontiguousarray(color_image[..., ::-1])  # BEV/cv2 expect BGR

        # ground-truth screen position of each of our 2 people, used to match
        # BEV's detections (order not guaranteed) back to person 0/1. Both
        # coordinates, not just x: orbiting the full 360 produces viewpoints
        # where one person stands behind the other and the x values coincide.
        screen = self.camera.cameras.transform_points_screen(
            frame_ctx['verts_c'].mean(1),
            image_size=torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]]).to(self.device))
        our_screen_xy = screen[:, :2].detach().cpu().numpy()  # [2,2]

        # Everything needed to reproduce this view's world -> camera map.
        # PyTorch3D stores it as a row-vector transform, x_cam = x_world @ R + T,
        # and we rendered (verts - center), so the full map is
        #   x_cam = (x_world - center) @ R + T
        cam = {
            'R': self.renderer.cameras.R[0].detach().clone(),
            'T': self.renderer.cameras.T[0].detach().clone(),
            'center': frame_ctx['center'].detach().clone(),
            'pitch': pitch, 'yaw': yaw, 'roll': roll, 'dist': dist,
            'radius': frame_ctx['radius'], 'base_dist': frame_ctx['base_dist'],
        }
        return bgr_image, our_screen_xy, cam

    def render_sample(self, sample, cam_override=None):
        """Single-view convenience wrapper, for the diagnostics that inspect one
        frame at a time (inspect_interx_frame.py, --calibrate, --report).

        cam_override replays a camera an earlier run stored on the sample:
        {'pitch','yaw','roll','dist'} from render_cam_euler / render_cam_dist.
        Without it a fresh six-view set is drawn and the first view is used, so
        the draw stays in step with what the full run would do.
        """
        ctx = self.prepare_frame(sample)
        if cam_override is None:
            pitch, yaw, roll, jitter = sample_camera_params(self.rng)[0]
            cam_params = (pitch, yaw, roll, ctx['base_dist'] * jitter)
        else:
            cam_params = (float(cam_override['pitch']), float(cam_override['yaw']),
                          float(cam_override['roll']), float(cam_override['dist']))
        bgr, xy, cam = self.render_view(ctx, cam_params)
        cam['J0'] = ctx['J0']
        return bgr, xy, cam

    @torch.no_grad()
    def pgt_to_camera_frame(self, sample, cam, frame_fix=None, J0=None):
        """Express the mocap-world-frame ground truth in BEV's camera frame.

        Two composed transforms:
          1. world -> PyTorch3D camera frame, from the camera we just rendered
             with. In the column-vector convention SMPL parameters use,
             x_cam = Rc @ x_world + t, with Rc = R^T and t = T - center @ R.
          2. PyTorch3D camera frame -> BEV's frame, the constant BEV_FRAME_FIX.
        Composing gives x_bev = (F @ Rc) x_world + F @ t.

        Rewriting SMPL parameters under a rigid transform x -> A x + b uses the
        fact that the root rotation acts about the rest-pose pelvis J0:
            verts = G (M - J0) + J0 + transl
        Substituting and matching the same form gives
            G'      = A G
            transl' = A (J0 + transl) + b - J0
        which is the same pelvis-aware bookkeeping
        llib/methods/hhc_diffusion/train_module.py::prep_translation does.
        """
        go = torch.from_numpy(sample['pgt_smplx_global_orient']).float().to(self.device)  # [2,3]
        transl = torch.from_numpy(sample['pgt_smplx_transl']).float().to(self.device)     # [2,3]

        R, T, center = cam['R'], cam['T'], cam['center']
        F = torch.from_numpy(BEV_FRAME_FIX if frame_fix is None else frame_fix).to(self.device)
        A = F @ R.transpose(0, 1)
        b = F @ (T - center @ R)

        # rest-pose pelvis of each person, computed once per frame by
        # prepare_frame rather than once per view -- it depends only on betas
        J0 = J0 if J0 is not None else cam['J0']  # [2,3]

        G_cam = torch.einsum('mn,bnl->bml', A, axis_angle_to_matrix(go))
        transl_cam = torch.einsum('mn,bn->bm', A, J0 + transl) + b - J0

        return {
            'pgt_smplx_global_orient_cam': matrix_to_axis_angle(G_cam).cpu().numpy().astype(np.float32),
            'pgt_smplx_transl_cam': transl_cam.cpu().numpy().astype(np.float32),
            'render_cam_R': R.cpu().numpy().astype(np.float32),
            'render_cam_T': T.cpu().numpy().astype(np.float32),
            'render_cam_center': center.cpu().numpy().astype(np.float32),
            'render_cam_euler': np.array([cam['pitch'], cam['yaw'], cam['roll']], dtype=np.float32),
            'render_cam_dist': np.float32(cam['dist']),
        }

    def match_detections(self, our_screen_xy, bev_pj2d_org):
        """Return [det_for_person0, det_for_person1], matching on 2D screen
        position rather than x alone.

        x alone was enough while the camera stayed within +-150 degrees of the
        front, where the pair is always seen side by side. Orbiting the full 360
        degrees adds viewpoints where one person occludes the other, their x
        coordinates coincide, and the match becomes a coin flip -- measured
        consequence on exactly those frames: layout error p90 of 167 degrees,
        i.e. the person0->person1 offset vector reversed. Use separation() to
        reject the frames where even 2D is ambiguous.
        """
        bev_xy = bev_pj2d_org[:, :, :2].mean(axis=1)  # [2, 2]
        cost = np.linalg.norm(our_screen_xy[:, None, :] - bev_xy[None, :, :], axis=-1)
        return [0, 1] if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else [1, 0]

    @staticmethod
    def match_detections_x(our_screen_xy, bev_pj2d_org):
        """The original x-only rule, kept so --ab-render can quantify what the
        2D rule buys."""
        bev_x = bev_pj2d_org[:, :, 0].mean(axis=1)
        c = np.abs(our_screen_xy[:, 0][:, None] - bev_x[None, :])
        return [0, 1] if c[0, 0] + c[1, 1] <= c[0, 1] + c[1, 0] else [1, 0]

    @staticmethod
    def separation(our_screen_xy):
        """Screen-space distance between the two people, in pixels."""
        return float(np.linalg.norm(our_screen_xy[0] - our_screen_xy[1]))

    def bev_to_smplx(self, bev_data, det_idx_for_person):
        """Mirrors llib/data/preprocess/hi4d.py::process_bev, per person."""
        out = {
            'bev_smplx_global_orient': [], 'bev_smplx_body_pose': [],
            'bev_smplx_betas': [], 'bev_smplx_scale': [], 'bev_smplx_transl': [],
        }

        for det_idx in det_idx_for_person:
            smpl_betas_scale = bev_data['smpl_betas'][det_idx]
            smpl_betas = smpl_betas_scale[:10]
            smpl_scale = smpl_betas_scale[-1]
            smpl_body_pose = bev_data['smpl_thetas'][det_idx][3:]
            smpl_global_orient = bev_data['smpl_thetas'][det_idx][:3]
            cam_trans = bev_data['cam_trans'][det_idx]

            if smpl_scale > 0.8:
                betas_scale = self.shape_converter_smil.forward(
                    torch.from_numpy(smpl_betas).unsqueeze(0).float())
            else:
                betas_scale = self.shape_converter_smpla.forward(
                    torch.from_numpy(smpl_betas_scale).unsqueeze(0).float())
            smplx_betas = betas_scale[0, :10].numpy()
            smplx_scale = betas_scale[0, 10].numpy()

            body_pose = smpl_body_pose[:63]
            global_orient = smpl_global_orient

            h_global_orient = torch.from_numpy(global_orient).float().unsqueeze(0)
            h_body_pose = torch.from_numpy(body_pose).float().unsqueeze(0)
            h_betas_scale = torch.from_numpy(
                np.concatenate((smplx_betas, smplx_scale[None]), axis=0)
            ).float().unsqueeze(0)

            body = self.bev_body_model(
                global_orient=h_global_orient, body_pose=h_body_pose, betas=h_betas_scale)
            root_trans = body.joints.detach()[:, 0, :]
            transl = -root_trans[0].numpy() + cam_trans

            out['bev_smplx_global_orient'].append(global_orient.astype(np.float32))
            out['bev_smplx_body_pose'].append(body_pose.astype(np.float32))
            out['bev_smplx_betas'].append(smplx_betas.astype(np.float32))
            out['bev_smplx_scale'].append(np.float32(smplx_scale))
            out['bev_smplx_transl'].append(transl.astype(np.float32))

        return {
            'bev_smplx_global_orient': np.stack(out['bev_smplx_global_orient']),
            'bev_smplx_body_pose': np.stack(out['bev_smplx_body_pose']),
            'bev_smplx_betas': np.stack(out['bev_smplx_betas']),
            'bev_smplx_scale': np.stack(out['bev_smplx_scale']),
            'bev_smplx_transl': np.stack(out['bev_smplx_transl']),
        }

    def process_frame(self, base, vis=None):
        """Turn one mocap frame into NUM_VIEWS conditioned training samples.

        Each view is a shallow copy of the frame's dict with only the
        view-dependent keys overwritten. That is deliberate: pickle memoises
        shared ndarrays and they stay shared after loading, so the six views cost
        one copy of pgt_smplx_body_pose, not six. The flip side is that nothing
        downstream may mutate a sample's arrays in place or rebind them per
        sample -- depenetrate_interx.py does exactly that, so it has to run
        against processed.pkl before this expansion, never after.
        """
        ctx = self.prepare_frame(base)
        frame_key = base['imgname']
        views = []

        for k, (pitch, yaw, roll, jitter) in enumerate(sample_camera_params(self.rng)):
            s = dict(base)
            # frame-level id, kept because imgname is now per view: the
            # penetration exclude list and every per-frame diagnostic key on this
            s['frame_key'] = frame_key
            s['view_idx'] = k
            s['imgname'] = f'{frame_key}_c{k}'
            # the 'InterX' substring is what llib/data/single.py dispatches on
            s['imgpath'] = f"InterX/{s['imgname']}.png"

            bgr, our_screen_xy, cam = self.render_view(
                ctx, (pitch, yaw, roll, ctx['base_dist'] * jitter))

            # The camera-frame ground truth only depends on the camera we drew,
            # so write it whether or not BEV detects anything below. With
            # allow_missing_bev=True those views still train, they just carry no
            # conditioning signal.
            s.update(self.pgt_to_camera_frame(base, cam, J0=ctx['J0']))

            sep = self.separation(our_screen_xy)
            bev_out, pairing = None, None
            if sep < MIN_SEPARATION_PX:
                # Projection degenerate: the two people land on nearly the same
                # spot, so even a correct BEV detection could not be assigned to
                # person 0/1 with any confidence. Bail out before spending a BEV
                # forward -- a guessed pairing enters training as
                # confident-but-wrong conditioning, strictly worse than the
                # missing conditioning recorded here.
                s['information_missing'] = True
            else:
                bev_out = self.bev_model(bgr)
                if bev_out is None or bev_out['cam_trans'].shape[0] != 2:
                    s['information_missing'] = True
                else:
                    pairing = self.match_detections(our_screen_xy, bev_out['pj2d_org'])
                    s.update(self.bev_to_smplx(bev_out, pairing))
                    # Reset explicitly rather than relying on stage 1's False: on a
                    # re-run (e.g. a different seed, to retry failures) a view that
                    # failed before but succeeds now would otherwise keep a stale
                    # information_missing=True and be filtered out despite having
                    # real bev_smplx_* values.
                    s['information_missing'] = False

            if vis is not None:
                vis.maybe_dump(s, bgr, our_screen_xy, bev_out, pairing, sep)
            views.append(s)

        return views


class VisSampler:
    """Sample a few BEV inputs to disk so the run leaves something inspectable.

    The pipeline keeps BEV resident in-process and consumes its output dict
    directly, so nothing is written for any of the ~286k views -- which is what
    makes the scale affordable (rendering them all the way CHI3D's `bev` CLI does
    would be roughly 570GB against 198GB free). But with no trace at all there is
    no way to confirm afterwards how BEV actually did, so keep a small sample.

    Successes are sampled by hashing frame_key, so a resumed or repeated run
    keeps the same ones rather than accumulating a biased pile. Failures get
    their own quota: judging BEV from successes alone says nothing about when it
    breaks, and 'when it breaks' is the number that matters here.
    """

    def __init__(self, out_dir, every=500, max_fail=200):
        self.out_dir = out_dir
        self.every = every
        self.max_fail = max_fail
        self.n_fail = 0
        self.rows = []
        os.makedirs(out_dir, exist_ok=True)

    def maybe_dump(self, sample, bgr, our_screen_xy, bev_out, pairing, sep):
        failed = bool(sample['information_missing'])
        if failed:
            if self.n_fail >= self.max_fail:
                return
            self.n_fail += 1
        elif self.every <= 0 or (hash(sample['frame_key']) % self.every) != 0:
            return

        euler = sample['render_cam_euler']
        _dump_overlay(osp.join(self.out_dir, f"{sample['imgname']}.png"),
                      bgr, our_screen_xy, bev_out, pairing, sep)
        n_det = 0 if bev_out is None else int(bev_out['cam_trans'].shape[0])
        self.rows.append((sample['frame_key'], sample['view_idx'], n_det, sep,
                          float(euler[1]), float(euler[0]), float(sample['render_cam_dist'])))

    def write_manifest(self):
        if not self.rows:
            return
        fn = osp.join(self.out_dir, 'manifest.csv')
        with open(fn, 'w') as f:
            f.write('frame_key,view_idx,n_det,separation_px,yaw,pitch,dist\n')
            for r in self.rows:
                f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{r[5]:.1f},{r[6]:.3f}\n')
        guru.info(f'wrote {len(self.rows)} overlays + {fn}')


CANDIDATE_FRAME_FIXES = {
    'identity': np.eye(3, dtype=np.float32),
    'flip_x (180 deg about X)': np.diag([1.0, -1.0, -1.0]).astype(np.float32),
    'flip_y (180 deg about Y)': np.diag([-1.0, 1.0, -1.0]).astype(np.float32),
    'flip_z (180 deg about Z)': np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
}


def calibrate(processed_data_folder, num_takes, seed):
    """Measure the constant rotation between our PyTorch3D render camera frame
    and the frame BEV reports in, so BEV_FRAME_FIX can be set from data instead
    of guessed from documentation.

    For each candidate constant F we express the ground truth as F @ (world ->
    camera) and compare it to BEV's estimate two ways:
      - orientation: the residual rotation angle between the two global
        orientations, per person;
      - layout: the angle between the person0 -> person1 offset vectors.
    The right F is the one where both collapse towards zero. Whatever is left
    over is BEV's own estimation error, so the residual spread doubles as an
    accuracy readout.
    """
    from scipy.spatial.transform import Rotation as Rot

    processed = pickle.load(open(osp.join(processed_data_folder, 'processed.pkl'), 'rb'))
    take_ids = list(processed.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(take_ids)
    take_ids = take_ids[:num_takes]

    proc = InterXBevProcessor(seed=seed)

    per_fix_orient = {k: [] for k in CANDIDATE_FRAME_FIXES}
    per_fix_layout = {k: [] for k in CANDIDATE_FRAME_FIXES}
    n_ok, n_total = 0, 0

    for take_id in tqdm(take_ids):
        for sample in processed[take_id]:
            n_total += 1
            bgr_image, our_screen_xy, cam = proc.render_sample(sample)
            bev_out = proc.bev_model(bgr_image)
            if bev_out is None or bev_out['cam_trans'].shape[0] != 2:
                continue
            pairing = proc.match_detections(our_screen_xy, bev_out['pj2d_org'])
            bev = proc.bev_to_smplx(bev_out, pairing)
            n_ok += 1

            R_bev = Rot.from_rotvec(bev['bev_smplx_global_orient'])
            t_bev = bev['bev_smplx_transl'][1] - bev['bev_smplx_transl'][0]

            for name, F in CANDIDATE_FRAME_FIXES.items():
                gt = proc.pgt_to_camera_frame(sample, cam, frame_fix=F)
                R_gt = Rot.from_rotvec(gt['pgt_smplx_global_orient_cam'])
                for i in range(2):
                    per_fix_orient[name].append(
                        np.degrees((R_bev[i] * R_gt[i].inv()).magnitude()))
                t_gt = gt['pgt_smplx_transl_cam'][1] - gt['pgt_smplx_transl_cam'][0]
                cos = np.dot(t_gt, t_bev) / (np.linalg.norm(t_gt) * np.linalg.norm(t_bev) + 1e-9)
                per_fix_layout[name].append(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    print(f'\n=== frame calibration on {n_ok}/{n_total} frames with a valid BEV detection ===')
    print(f'{"candidate F":28s} {"orient p50":>11s} {"orient p90":>11s} {"layout p50":>11s} {"layout p90":>11s}')
    best, best_score = None, 1e9
    for name in CANDIDATE_FRAME_FIXES:
        o = np.array(per_fix_orient[name]); l = np.array(per_fix_layout[name])
        print(f'{name:28s} {np.percentile(o,50):10.1f}d {np.percentile(o,90):10.1f}d '
              f'{np.percentile(l,50):10.1f}d {np.percentile(l,90):10.1f}d')
        score = np.percentile(o, 50) + np.percentile(l, 50)
        if score < best_score:
            best, best_score = name, score
    print(f'\nbest candidate: {best}')
    print(f'currently hardcoded BEV_FRAME_FIX =\n{BEV_FRAME_FIX}')


def bev_report(processed_data_folder, num_takes, seed, dump_dir=None, dump_n=12,
               min_sep=None):
    """Health check on the BEV conditioning: how often it is usable, how far off
    it is when it is, and whether the person0/person1 assignment holds up.

    Writes nothing to the dataset. Re-run this after any change to the camera
    sampler or the renderer -- in particular after the six-view change, since
    orbiting the full 360 degrees adds the occluded viewpoints that MIN_SEPARATION_PX
    exists to reject, and the threshold should be re-checked against the table
    this prints rather than carried over on faith.

    Columns:
      unusable / degen / det=1 / det>=3
          why a frame yields no conditioning. `degen` is the MIN_SEPARATION_PX
          bail-out, the rest is BEV failing to find exactly two people (det=1 is
          the two being merged into one).
      orient / layout
          error of BEV's estimate against the camera-frame ground truth: the
          residual rotation per person, and the angle between the
          person0 -> person1 offset vectors.
      pairing table
          flip rate (layout error > 90 deg, i.e. the two people swapped) binned
          by how far apart they are on screen, under both the 2D matching rule
          and the older x-only one. This is what MIN_SEPARATION_PX is set from.
    """
    from scipy.spatial.transform import Rotation as Rot

    processed = pickle.load(open(osp.join(processed_data_folder, 'processed.pkl'), 'rb'))
    take_ids = list(processed.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(take_ids)
    take_ids = take_ids[:num_takes]

    proc = InterXBevProcessor(seed=seed)

    # Overriding the guard to 0 is how you re-derive the threshold: with it
    # active the sub-threshold bins are empty by construction, so the table can
    # only ever confirm the value already in use, never correct it.
    guard = MIN_SEPARATION_PX if min_sep is None else min_sep

    ndet, orient, layout = [], [], []
    sep_all, sep_ok, l2d, lx = [], [], [], []
    n_dumped = 0
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)

    for take_id in tqdm(take_ids):
        for sample in processed[take_id]:
            bgr, our_screen_xy, cam = proc.render_sample(sample)
            sep = proc.separation(our_screen_xy)
            sep_all.append(sep)

            if sep < guard:
                ndet.append(-1)                      # same bail-out process_sample does
                bev_out, pairing = None, None
            else:
                bev_out = proc.bev_model(bgr)
                n = 0 if bev_out is None else int(bev_out['cam_trans'].shape[0])
                ndet.append(n)
                pairing = (proc.match_detections(our_screen_xy, bev_out['pj2d_org'])
                           if n == 2 else None)

            if pairing is not None:
                gt = proc.pgt_to_camera_frame(sample, cam)
                R_gt = Rot.from_rotvec(gt['pgt_smplx_global_orient_cam'])
                t_gt = gt['pgt_smplx_transl_cam'][1] - gt['pgt_smplx_transl_cam'][0]

                def _layout(pr):
                    b = proc.bev_to_smplx(bev_out, pr)
                    t = b['bev_smplx_transl'][1] - b['bev_smplx_transl'][0]
                    c = np.dot(t_gt, t) / (np.linalg.norm(t_gt) * np.linalg.norm(t) + 1e-9)
                    return np.degrees(np.arccos(np.clip(c, -1, 1))), b

                l, bev = _layout(pairing)
                R_bev = Rot.from_rotvec(bev['bev_smplx_global_orient'])
                orient.extend(np.degrees((R_bev[i] * R_gt[i].inv()).magnitude()) for i in range(2))
                layout.append(l)
                sep_ok.append(sep)
                l2d.append(l)
                lx.append(_layout(proc.match_detections_x(
                    our_screen_xy, bev_out['pj2d_org']))[0])

            if dump_dir is not None and n_dumped < dump_n:
                _dump_overlay(osp.join(dump_dir, f"{sample['imgname']}.png"),
                              bgr, our_screen_xy, bev_out, pairing, sep)
                n_dumped += 1

    d = np.array(ndet)
    o = np.array(orient) if orient else np.array([np.nan])
    l = np.array(layout) if layout else np.array([np.nan])
    print(f'\n=== BEV conditioning report: {len(d)} frames from {len(take_ids)} takes ===\n')
    print(f'{"unusable":>9s} {"degen":>7s} {"det=1":>7s} {"det>=3":>7s} '
          f'{"orient p50":>11s} {"orient p90":>11s} {"layout p50":>11s} {"layout p90":>11s}')
    print(f'{(d != 2).mean()*100:8.1f}% {(d == -1).mean()*100:6.1f}% {(d == 1).mean()*100:6.1f}% '
          f'{(d >= 3).mean()*100:6.1f}% '
          f'{np.percentile(o,50):10.1f}d {np.percentile(o,90):10.1f}d '
          f'{np.percentile(l,50):10.1f}d {np.percentile(l,90):10.1f}d')

    sep_all = np.array(sep_all); sep_ok = np.array(sep_ok)
    l2d = np.array(l2d); lx = np.array(lx)
    print('\npairing: flip rate (layout error > 90 deg = person0/person1 swapped),')
    print(f'binned by screen-space distance between the two people. '
          f'Guard active in this run: < {guard:.0f}px.')
    print(f'{"separation px":>16s} {"n":>7s} {"x-only flip":>12s} {"2D flip":>9s} '
          f'{"x-only l p50":>13s} {"2D l p50":>10s}')
    # fine-grained around the decision boundary: the six-view data put the
    # flip rate at 22.6% in 12-20px and ~1% above 20px, so the knee that sets
    # MIN_SEPARATION_PX lives inside that range
    edges = [0, 6, 9, 12, 14, 16, 18, 20, 25, 30, 40, 60, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sep_ok >= lo) & (sep_ok < hi)
        if not m.any():
            continue
        name = f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'{lo:.0f}+'
        print(f'{name:>16s} {int(m.sum()):7d} {(lx[m] > 90).mean()*100:11.1f}% '
              f'{(l2d[m] > 90).mean()*100:8.1f}% '
              f'{np.percentile(lx[m],50):12.1f}d {np.percentile(l2d[m],50):9.1f}d')
    if len(l2d):
        print(f'{"ALL":>16s} {len(l2d):7d} {(lx > 90).mean()*100:11.1f}% {(l2d > 90).mean()*100:8.1f}%')
    print('separation percentiles over all frames: ' + '  '.join(
        f'p{q}={np.percentile(sep_all, q):.0f}' for q in (1, 5, 25, 50, 75, 95)) + ' px')
    if dump_dir is not None:
        print(f'\nwrote {n_dumped} overlays to {dump_dir}')


def _dump_overlay(path, bgr, our_screen_xy, bev_out, pairing, sep):
    """Save the exact image BEV saw, with our ground-truth screen positions and
    BEV's joints drawn on top, coloured by the identity our matching assigned.

    Reading it: the cross and the skeleton of the same colour should sit on the
    same person. If they are swapped, match_detections got the pairing wrong --
    which is the failure MIN_SEPARATION_PX is meant to catch, and it does not
    show up in information_missing.
    """
    import cv2
    img = bgr.copy()
    # BGR: person0 blue #2a78d6, person1 orange #eb6834. Blue/orange rather than
    # the obvious red/green because these overlays end up in figures: under
    # deuteranopia the old red/green pair separates by an OKLab dE of only ~8.5,
    # against ~37 for this pair, and identity here is carried by colour alone.
    colors = [(214, 120, 42), (52, 104, 235)]
    if pairing is not None:
        for person, det in enumerate(pairing):
            for x, y in bev_out['pj2d_org'][det][:, :2]:
                cv2.circle(img, (int(x), int(y)), 2, colors[person], -1)
    for person, (x, y) in enumerate(our_screen_xy):
        cv2.drawMarker(img, (int(x), int(y)), colors[person], cv2.MARKER_CROSS, 26, 2)
    n = 'skipped' if pairing is None and bev_out is None else (
        'det!=2' if pairing is None else '2')
    cv2.putText(img, f'sep {sep:.0f}px  det {n}', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def atomic_save(obj, out_fn):
    """Write to a temp file then rename, so a crash/kill mid-write never
    leaves processed.pkl truncated/corrupted -- the previous good version
    stays in place until the new one is fully written."""
    tmp_fn = out_fn + '.tmp'
    with open(tmp_fn, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_fn, out_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    parser.add_argument('--in-fn', default='processed.pkl',
                        help='stage-1 output to read. Never written to.')
    parser.add_argument('--out-fn', default='processed_mv.pkl',
                        help='where the six-view result goes. Kept separate from --in-fn so the '
                             'stage-1 source stays clean and running twice cannot produce 36 '
                             'views per frame. Note --in-fn is NOT usable as a single-view '
                             'control: its bev_* fields are still the all-zero template. The '
                             'control is a previously BEV-processed single-view file, e.g. '
                             'processed_sv_stride20_oldcam.pkl.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit-takes', type=int, default=-1, help='only process the first N takes, for a quick dry run')
    parser.add_argument('--save-every', type=int, default=500, help='checkpoint the output to disk every N takes, so a crash mid-run does not lose everything')
    parser.add_argument('--vis-dir', default=None,
                        help='sample a few BEV inputs here with our GT positions and BEV joints '
                             'drawn on top, so the run leaves something inspectable. Defaults to '
                             '<processed-data-folder>/diagnostics/bev_vis_mv; pass "" to disable.')
    parser.add_argument('--vis-every', type=int, default=500,
                        help='dump one successful view in every N (sampled by hashing frame_key, '
                             'so a resumed run keeps the same ones). 0 disables success sampling.')
    parser.add_argument('--vis-max-fail', type=int, default=200,
                        help='dump up to this many failed views. Failures get their own quota '
                             'because judging BEV from successes alone says nothing about when '
                             'it breaks.')
    parser.add_argument('--report', type=int, default=0, metavar='N',
                        help='do not write anything: render N random takes, run BEV on them and report how usable the conditioning is (detection rate, orientation/layout error, pairing flip rate vs screen separation). Re-run after any camera or renderer change.')
    parser.add_argument('--report-dump-dir', default=None, help='with --report, save this many BEV-input overlays here for eyeballing')
    parser.add_argument('--report-dump-n', type=int, default=12)
    parser.add_argument('--report-min-sep', type=float, default=None,
                        help='override MIN_SEPARATION_PX for this report. Pass 0 to disable the '
                             'guard and see the flip rate in the bins it normally removes -- '
                             'required to re-derive the threshold rather than just confirm it.')
    parser.add_argument('--calibrate', type=int, default=0, metavar='N',
                        help='do not write anything: render N random takes, run BEV on them, and report which constant BEV_FRAME_FIX best aligns our camera frame with BEV s output')
    args = parser.parse_args()

    if args.report > 0:
        bev_report(args.processed_data_folder, args.report, args.seed,
                   dump_dir=args.report_dump_dir, dump_n=args.report_dump_n,
                   min_sep=args.report_min_sep)
        return

    if args.calibrate > 0:
        calibrate(args.processed_data_folder, args.calibrate, args.seed)
        return

    in_fn = osp.join(args.processed_data_folder, args.in_fn)
    out_fn = osp.join(args.processed_data_folder, args.out_fn)
    progress_fn = out_fn + '.progress'

    source = pickle.load(open(in_fn, 'rb'))
    take_ids = list(source.keys())
    if args.limit_takes > 0:
        take_ids = take_ids[:args.limit_takes]

    # Resume from the output rather than mutating the input in place: finished
    # takes are whatever is already in out_fn, cross-checked against a progress
    # sidecar so a take interrupted mid-write is redone rather than half kept.
    processed = {}
    done_take_ids = set()
    if osp.exists(out_fn) and osp.exists(progress_fn):
        processed = pickle.load(open(out_fn, 'rb'))
        done_take_ids = pickle.load(open(progress_fn, 'rb')) & set(processed)
        guru.info(f'Resuming: {len(done_take_ids)} takes already in {out_fn}, skipping those.')
    elif osp.exists(out_fn):
        guru.warning(f'{out_fn} exists but {progress_fn} does not -- starting over and '
                     f'overwriting it. Move it aside first if you meant to keep it.')

    remaining_take_ids = [t for t in take_ids if t not in done_take_ids]
    guru.info(f'{len(remaining_take_ids)}/{len(take_ids)} takes left to process, '
              f'{NUM_VIEWS} views each.')

    vis_dir = args.vis_dir
    if vis_dir is None:
        vis_dir = osp.join(args.processed_data_folder, 'diagnostics', 'bev_vis_mv')
    vis = VisSampler(vis_dir, args.vis_every, args.vis_max_fail) if vis_dir else None

    proc = InterXBevProcessor(seed=args.seed)

    n_total, n_missing = 0, 0
    for i, take_id in enumerate(tqdm(remaining_take_ids)):
        views = []
        for sample in source[take_id]:
            views.extend(proc.process_frame(sample, vis=vis))
        processed[take_id] = views
        n_total += len(views)
        n_missing += sum(v['information_missing'] for v in views)
        done_take_ids.add(take_id)

        if (i + 1) % args.save_every == 0:
            atomic_save(processed, out_fn)
            atomic_save(done_take_ids, progress_fn)
            guru.info(f'Checkpointed at {i+1}/{len(remaining_take_ids)} takes this run '
                      f'({len(done_take_ids)}/{len(take_ids)} total).')

    guru.info(f'Produced {n_total} views this run, {n_missing} '
              f'({n_missing/max(n_total,1)*100:.1f}%) with no usable BEV conditioning '
              f'(information_missing=True: either BEV found != 2 people, or the two '
              f'projected closer than {MIN_SEPARATION_PX:.0f}px apart)')

    atomic_save(processed, out_fn)
    atomic_save(done_take_ids, progress_fn)
    guru.info(f'Saved {out_fn}')
    if vis is not None:
        vis.write_manifest()
    if len(done_take_ids) >= len(take_ids):
        os.remove(progress_fn)
        guru.info('All takes processed, removed progress sidecar file.')
    guru.info(f'Next: point datasets.interx.processed_fn at {args.out_fn}, and delete any '
              f'cached {{split}}_diffusion.pkl so the loader rebuilds from it.')


if __name__ == '__main__':
    main()
