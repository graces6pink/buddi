"""
Paired conditional-generation evaluation for a BEV-conditioned BUDDI checkpoint.

This fills a real gap in the repo. Existing evaluation covers:
  - Trainer.validate() / EvalModule -> single-step denoising at t=50 only
    (that is the number in the checkpoint filename), and
  - evaluation/eval.py + compare_samples_*.py -> distribution-level metrics
    (FID / diversity / penetration) for UNCONDITIONAL models.
Nothing runs the full denoising loop WITH the BEV conditioning and compares the
result against the paired ground truth. train_module.py::single_validation_step
does have a conditional sampling branch, but it only renders tensorboard images
and accumulates no metrics at all.

What this script does, per checkpoint:
  BEV condition -> full DDIM sampling loop -> SMPL-X -> metrics vs paired mocap GT
and, as a control, the same with the conditioning nulled out (classifier-free
guidance's null value), so you can see how much the BEV condition actually buys.

All sampling and all metric implementations are reused from the existing code:
  - TrainModule.get_guidance_params / preprocess_batch / cast_smpl /
    get_gt_params / get_smpl / sample_from_model   (train_module.py)
  - setup_diffusion_module                          (evaluation/utils.py)
  - PointError metrics via build_metric             (llib/utils/metrics/build.py)
  - ContactMap.get_full_heatmap, thresholded at 0.013m, identical to
    hhcs_optimization/evaluation/flickrci3ds_eval.py

Three metrics, matching what the paper reports:
  pa_mpjpe_h0 / pa_mpjpe_h1   Table 3's PER PERSON columns
  pa_mpjpe_h0h1               Table 3's JOINT column (both people aligned once)
  pcc@r                       fraction of the GT contact pairs brought within r

Point metrics are built directly with build_metric rather than through
EvalModule.forward, which dispatches per-person metrics by name and silently
reuses the previous metric's points for any name it has no branch for
(eval_module.py:165-170). Building the metric objects directly avoids that trap
and keeps the per-sample arrays around for the .npz dumps.

Usage:
    python llib/methods/hhc_diffusion/evaluation/eval_conditional.py \
      --exp-cfg demo/diffusion/training/interx_cond_bev_camfix/config.yaml \
      --checkpoint-dir demo/diffusion/training/interx_cond_bev_camfix/checkpoints \
      --sweep-n 10 \
      --output-folder demo/diffusion/eval/camfix_cond_sweep \
      --skip-steps 10 --batch-size 256
"""

import argparse
import csv
import json
import os
import os.path as osp
import pickle
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as guru
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from llib.data.build import build_datasets
from llib.defaults.main import config as default_config, merge as merge_configs
from llib.methods.hhc_diffusion.evaluation.utils import setup_diffusion_module
from llib.utils.metrics.build import build_metric
from llib.utils.threed.distance import ContactMap

ESSENTIALS_HOME = 'essentials'
REGION_TO_VERTEX_PATH = osp.join(
    ESSENTIALS_HOME, 'contact/flickrci3ds_r75_rid_to_smplx_vid.pkl')
J14_REGRESSOR_PATH = osp.join(
    ESSENTIALS_HOME, 'body_model_utils/joint_regressors/SMPLX_to_J14.pkl')

# same binary contact threshold as flickrci3ds_eval.py:251
CONTACT_THRESHOLD = 0.013

# checkpoint filename layout, see Logger.get_checkpoint_fn:
# <timestamp>__<epoch>__<batch_idx>__<val_error>.pt
CKPT_RE = re.compile(r'^(?P<ts>[\d_\-]+)__(?P<step>\d+)__(?P<batch>\d+)__(?P<val>[\d.]+)\.pt$')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp-cfg', dest='exp_cfgs', nargs='+', required=True,
                        help='Config the checkpoint was trained with, e.g. '
                             '<run folder>/config.yaml')
    parser.add_argument('--exp-opts', dest='exp_opts', nargs='*', default=[])
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Sweep every Nth checkpoint in this folder.')
    parser.add_argument('--checkpoints', nargs='*', default=[],
                        help='Explicit checkpoint paths. Overrides --checkpoint-dir.')
    parser.add_argument('--sweep-n', type=int, default=10,
                        help='How many checkpoints to pick from --checkpoint-dir '
                             '(evenly spaced by training step; the lowest-val-loss '
                             'checkpoint is always included).')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, default='interx')
    parser.add_argument('--dataset-split', type=str, default='val')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--skip-steps', type=int, default=10,
                        help='DDIM stride. 10 matches the validation loop.')
    parser.add_argument('--max-batches', type=int, default=-1,
                        help='Cap the number of batches (debugging).')
    parser.add_argument('--contact-chunk', type=int, default=64,
                        help='Sub-batch size for the 75x75 contact heatmap.')
    parser.add_argument('--joints', choices=['model', 'j14'], default='model',
                        help="Joint set for the *_mpjpe metrics. 'model' = the "
                             "body model's own joints (coco25 with these configs), "
                             "'j14' = LSP-14 via SMPLX_to_J14.pkl (needs the "
                             "paper's essentials bundle).")
    parser.add_argument('--seed', type=int, default=238492)
    parser.add_argument('--no-uncond-control', action='store_true',
                        help='Skip the nulled-conditioning control run.')
    parser.add_argument('--allow-missing-bev', action='store_true',
                        help='Keep samples whose BEV estimate failed. Off by '
                             'default: they carry all-zero bev_* placeholders and '
                             'get their guidance nulled anyway, so including them '
                             'in a *conditional* evaluation is misleading.')

    cmd_args = parser.parse_args()
    cfg = merge_configs(cmd_args, default_config)
    return cfg, cmd_args


def prepare_cfg(cfg, cmd_args):
    """Force the settings a conditional evaluation needs, loudly."""

    # only the evaluation split, and never the (huge) training set
    for split in ['train', 'val', 'test']:
        cfg.datasets[f'{split}_names'] = []
    cfg.datasets[f'{cmd_args.dataset_split}_names'] = [cmd_args.dataset_name]
    if cmd_args.dataset_split == 'train':
        cfg.datasets.train_composition = [1.0]

    # training runs with mirror/swap/noise augmentation on; evaluating through it
    # would measure the augmentation as much as the model
    cfg.datasets.augmentation.use = False
    cfg.datasets.processing.use = False
    cfg.datasets.processing.load_image = False

    if cmd_args.dataset_name == 'interx':
        cfg.datasets.interx.allow_missing_bev = cmd_args.allow_missing_bev

    cfg.batch_size = cmd_args.batch_size

    # keep Logger (created inside setup_diffusion_module) away from the training
    # run folder -- it creates subfolders and writes a config snapshot
    cfg.logging.base_folder = cmd_args.output_folder
    cfg.logging.run = 'setup'

    return cfg


def select_checkpoints(cmd_args):
    """Return [(step, val_loss, path), ...] sorted by training step."""

    if cmd_args.checkpoints:
        out = []
        for path in cmd_args.checkpoints:
            match = CKPT_RE.match(osp.basename(path))
            step = int(match.group('step')) if match else -1
            val = float(match.group('val')) if match else float('nan')
            out.append((step, val, osp.abspath(path)))
        return sorted(out)

    assert cmd_args.checkpoint_dir is not None, \
        'pass either --checkpoints or --checkpoint-dir'

    found = []
    for fn in os.listdir(cmd_args.checkpoint_dir):
        match = CKPT_RE.match(fn)
        if match is None:
            continue
        found.append((int(match.group('step')), float(match.group('val')),
                      osp.abspath(osp.join(cmd_args.checkpoint_dir, fn))))
    assert found, f'no checkpoints matched in {cmd_args.checkpoint_dir}'
    found.sort()

    n = min(cmd_args.sweep_n, len(found))
    idxs = set(np.linspace(0, len(found) - 1, n).round().astype(int).tolist())
    idxs.add(int(np.argmin([v for _, v, _ in found])))  # always keep the best
    return [found[i] for i in sorted(idxs)]


class Metrics:
    """All the per-sample metric computations, in one place."""

    def __init__(self, cfg, device, contact_chunk=64, joints='model'):
        self.device = device
        self.contact_chunk = contact_chunk
        self.joints = joints

        # Which joint set the *_mpjpe metrics are computed on.
        #   'model' -- the body model's own joints. With the joint_mapper these
        #              configs use (smpl_to_openpose, coco25, no hands/face) that
        #              is 25 OpenPose body joints, the same points EvalModule
        #              uses during training. Root alignment defaults to joint 8
        #              (MidHip), i.e. the usual pelvis-relative MPJPE.
        #   'j14'   -- the 14 LSP joints via SMPLX_to_J14.pkl, matching
        #              flickrci3ds_eval.py. That regressor ships with the paper's
        #              essentials bundle and is not in every checkout, so it is
        #              opt-in rather than the default.
        self.j14_regressor = None
        if joints == 'j14':
            if not osp.exists(J14_REGRESSOR_PATH):
                raise FileNotFoundError(
                    f'--joints j14 needs {J14_REGRESSOR_PATH}, which is missing '
                    'from this checkout. Use --joints model instead.')
            j14 = pickle.load(open(J14_REGRESSOR_PATH, 'rb'), encoding='latin1')
            self.j14_regressor = torch.from_numpy(j14).to(device).float()
        # (no 'else' branch: the remaining metrics are all Procrustes-aligned, so
        # there is no root joint to pick)

        # Deliberately three metrics only, matching the paper's own reporting:
        #   pa_mpjpe_h{0,1}  -> Table 3's PER PERSON columns
        #   pa_mpjpe_h0h1    -> Table 3's JOINT column
        #   pcc              -> the contact metric the paper introduces
        # Earlier versions also computed root/scale-aligned and unaligned MPJPE,
        # v2v, and binary contact IoU/precision/recall/fscore. They are gone: on
        # this data a ground-truth frame has roughly one region pair in contact
        # out of 5625, so the binary 13mm IoU is near-chance and does not separate
        # models, while the alignment variants mostly moved together with
        # pa_mpjpe and made the tables harder to read.
        self.pa_mpjpe = build_metric(cfg.evaluation.pa_mpjpe)              # procrustes
        self.pairwise_pa_mpjpe = build_metric(cfg.evaluation.pairwise_pa_mpjpe)
        self.cmapper = ContactMap(region_to_vertex=REGION_TO_VERTEX_PATH).to(device)

        self.pcc_x = torch.from_numpy(np.arange(0.0, 1.0, 0.05)).to(device)

    def get_joints(self, smpl):
        if self.j14_regressor is not None:
            return torch.matmul(self.j14_regressor, smpl.vertices.detach())
        return smpl.joints.detach()

    def heatmap(self, v0, v1):
        """75x75 region-to-region min distance, chunked to bound memory."""
        out = []
        for i in range(0, v0.shape[0], self.contact_chunk):
            sl = slice(i, i + self.contact_chunk)
            out.append(self.cmapper.get_full_heatmap(v0[sl], v1[sl]))
        return torch.cat(out, dim=0)

    def __call__(self, est_smpl, tar_smpl):
        """Returns dict of per-sample numpy arrays."""

        out = {}

        est_verts = [est_smpl[i].vertices.detach() for i in range(2)]
        tar_verts = [tar_smpl[i].vertices.detach() for i in range(2)]
        est_j = [self.get_joints(est_smpl[i]) for i in range(2)]
        tar_j = [self.get_joints(tar_smpl[i]) for i in range(2)]

        # per person: Procrustes-aligned joint error (Table 3 PER PERSON)
        for i in range(2):
            out[f'pa_mpjpe_h{i}'] = self.pa_mpjpe(
                est_j[i].cpu().numpy(), tar_j[i].cpu().numpy()).mean(-1)

        # both people concatenated, aligned once (Table 3 JOINT). One alignment
        # for the pair means their relative orientation and spacing cannot be
        # absorbed by the fit, which is exactly what a two-person prior is for.
        est_both = torch.cat(est_j, dim=1).cpu().numpy()
        tar_both = torch.cat(tar_j, dim=1).cpu().numpy()
        out['pa_mpjpe_h0h1'] = self.pairwise_pa_mpjpe(est_both, tar_both).mean(-1)

        # PCC: of the region pairs the GT says are touching, what fraction did we
        # bring within distance x. Same construction as flickrci3ds_eval.py:258-263,
        # computed per sample so frames with no GT contact can be dropped rather
        # than poisoning the mean with a 0/0.
        est_heat = self.heatmap(est_verts[0], est_verts[1])
        tar_binary = self.heatmap(tar_verts[0], tar_verts[1]) < CONTACT_THRESHOLD

        pcc = np.full((est_heat.shape[0], len(self.pcc_x)), np.nan)
        for i in range(est_heat.shape[0]):
            if int(tar_binary[i].sum()) == 0:
                continue
            distances = est_heat[i][tar_binary[i]]
            hits = (distances[None] < self.pcc_x[:, None]).sum(1)
            pcc[i] = (hits.float() / distances.numel()).cpu().numpy()
        out['pcc'] = pcc

        return out


@torch.no_grad()
def evaluate_checkpoint(diffusion_module, loader, metrics, cmd_args, timesteps, mode):
    """mode: 'cond' (real BEV guidance) or 'uncond' (guidance nulled out)."""

    diffusion_module.eval()
    accumulator, n_dropped = {}, 0
    bs = diffusion_module.cfg.batch_size

    for batch_idx, batch in enumerate(tqdm(loader, desc=f'{mode}')):
        if 0 < cmd_args.max_batches <= batch_idx:
            break

        batch = {k: (v.to(cmd_args.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        # sampling_loop builds its timestep tensor with self.bs (train_module.py:640),
        # so a partial batch raises rather than just being wrong. DataLoader uses
        # drop_last, this is the belt-and-braces check.
        if batch['pgt_transl'].shape[0] != bs:
            n_dropped += batch['pgt_transl'].shape[0]
            continue

        # guidance has to be read off the raw batch, before preprocess_batch
        # overwrites it with the pgt params -- same order as single_validation_step
        if mode == 'cond':
            guidance = diffusion_module.get_guidance_params(
                batch, guidance_param_nc=0.0, guidance_all_nc=0.0,
                guidance_no_nc=1.0, clone=True)
        else:
            guidance = diffusion_module.get_guidance_params(
                batch, guidance_param_nc=0.0, guidance_all_nc=1.0,
                guidance_no_nc=0.0, clone=True)

        batch = diffusion_module.cast_smpl(diffusion_module.preprocess_batch(batch))
        target_smpls = diffusion_module.get_smpl(
            diffusion_module.get_gt_params(batch))

        # large log_freq => only x_starts['final'] is kept
        _, x_starts = diffusion_module.sample_from_model(
            timesteps, log_freq=10 ** 9, guidance_params=guidance)

        for key, value in metrics(x_starts['final'], target_smpls).items():
            accumulator.setdefault(key, []).append(value)

    if not accumulator:
        raise RuntimeError('no full batch was evaluated -- lower --batch-size')

    out = {k: np.concatenate(v, axis=0) for k, v in accumulator.items()}
    out['_n_dropped'] = n_dropped
    return out


def summarise(per_sample):
    """Per-sample arrays -> scalar summary. mpjpe-family converted to mm."""
    summary = {}
    for key, value in per_sample.items():
        if key.startswith('_'):
            continue
        if key == 'pcc':
            valid = ~np.isnan(value).any(axis=1)
            mean = value[valid].mean(axis=0) if valid.any() else np.full(value.shape[1], np.nan)
            for x, y in zip(np.arange(0.0, 1.0, 0.05), mean):
                summary[f'pcc@{x:.2f}'] = float(y)
            summary['pcc_n_samples_with_gt_contact'] = int(valid.sum())
            continue
        scale = 1000.0 if 'mpjpe' in key or 'v2v' in key else 1.0
        if np.isnan(value).any():
            valid = ~np.isnan(value)
            summary[key] = float(np.mean(value[valid]) * scale) if valid.any() else float('nan')
            summary[f'{key}_n_defined'] = int(valid.sum())
        else:
            summary[key] = float(np.mean(value) * scale)
    return summary


def write_curves(rows, output_folder):
    """One subplot per metric, cond vs uncond, against training step."""

    plot_metrics = ['pa_mpjpe_h0', 'pa_mpjpe_h1', 'pa_mpjpe_h0h1', 'pcc@0.10']
    plot_metrics = [m for m in plot_metrics if m in rows[0]]

    n = len(plot_metrics)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)

    for ax, metric in zip(axes.flatten(), plot_metrics):
        for mode, style in [('cond', '-o'), ('uncond', '--s')]:
            pts = sorted((r['step'], r[metric]) for r in rows if r['mode'] == mode)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        label=mode, markersize=4)
        ax.set_title(metric)
        ax.set_xlabel('training step')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.flatten()[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(osp.join(output_folder, 'curves.png'), dpi=120)
    plt.close(fig)


def write_pcc(rows, output_folder):
    pcc_keys = sorted([k for k in rows[0] if k.startswith('pcc@')],
                      key=lambda k: float(k.split('@')[1]))
    if not pcc_keys:
        return
    xs = [float(k.split('@')[1]) for k in pcc_keys]

    # pick by the JOINT column: it is the one that reflects how the two people
    # are arranged relative to each other, which is what the prior is for
    best = min((r for r in rows if r['mode'] == 'cond'),
               key=lambda r: r['pa_mpjpe_h0h1'])
    fig, ax = plt.subplots(figsize=(5, 4))
    for row in rows:
        if row['step'] != best['step']:
            continue
        ax.plot(xs, [row[k] for k in pcc_keys], '-o', markersize=3,
                label=f"{row['mode']} (step {row['step']})")
    ax.set_xlabel('distance threshold (m)')
    ax.set_ylabel('PCC')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(output_folder, 'pcc.png'), dpi=120)
    plt.close(fig)


def main(cfg, cmd_args):
    os.makedirs(cmd_args.output_folder, exist_ok=True)
    cfg = prepare_cfg(cfg, cmd_args)
    cmd_args.device = cfg.device

    checkpoints = select_checkpoints(cmd_args)
    guru.info(f'Evaluating {len(checkpoints)} checkpoint(s): '
              + ', '.join(f'step {s} (val {v})' for s, v, _ in checkpoints))

    # dataset once, reused for every checkpoint
    _, val_datasets = build_datasets(
        datasets_cfg=cfg.datasets,
        body_model_type=cfg.body_model.type,
        build_train=(cmd_args.dataset_split == 'train'),
    )
    dataset = val_datasets[cmd_args.dataset_name]
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers, pin_memory=False,
                        drop_last=True)
    n_used = (len(dataset) // cfg.batch_size) * cfg.batch_size
    guru.info(f'{cmd_args.dataset_name} [{cmd_args.dataset_split}]: '
              f'{len(dataset)} samples loaded, {n_used} used '
              f'({len(dataset) - n_used} dropped by drop_last), '
              f'allow_missing_bev={cmd_args.allow_missing_bev}')

    metrics = Metrics(cfg, cfg.device, contact_chunk=cmd_args.contact_chunk,
                      joints=cmd_args.joints)

    modes = ['cond'] if cmd_args.no_uncond_control else ['cond', 'uncond']
    diffusion_module, timesteps, rows = None, None, []

    for step, val_loss, ckpt_path in checkpoints:
        if diffusion_module is None:
            cmd_args.checkpoint_name = ckpt_path
            diffusion_module = setup_diffusion_module(cfg, cmd_args)
            # same schedule the validation loop uses (train_module.py:1227)
            timesteps = np.arange(
                1, diffusion_module.diffusion.num_timesteps, cmd_args.skip_steps)[::-1]
            guru.info(f'DDIM schedule: {len(timesteps)} steps '
                      f'({timesteps[0]} -> {timesteps[-1]})')
        else:
            # only the weights change between sweep points
            state = torch.load(ckpt_path)
            diffusion_module.model.load_state_dict(state['model'], strict=False)
            diffusion_module.model.eval()

        for mode in modes:
            # identical noise for every checkpoint, so a change in the curve is a
            # change in the model (same trick as diffusion_trainer.py:211)
            torch.manual_seed(cmd_args.seed)
            np.random.seed(cmd_args.seed)
            torch.cuda.manual_seed_all(cmd_args.seed)

            per_sample = evaluate_checkpoint(
                diffusion_module, loader, metrics, cmd_args, timesteps, mode)

            row = {'step': step, 'val_loss_from_filename': val_loss,
                   'mode': mode, 'checkpoint': ckpt_path,
                   'n_samples': int(len(per_sample['pa_mpjpe_h0h1']))}
            row.update(summarise(per_sample))
            rows.append(row)

            dump_folder = osp.join(cmd_args.output_folder, 'per_checkpoint',
                                   f'{step:010d}')
            os.makedirs(dump_folder, exist_ok=True)
            np.savez_compressed(
                osp.join(dump_folder, f'{mode}.npz'),
                **{k: v for k, v in per_sample.items() if not k.startswith('_')})

            guru.info(f"step {step} [{mode}] "
                      f"pa_mpjpe_h0h1={row['pa_mpjpe_h0h1']:.1f}mm "
                      f"pa_mpjpe_h0h1={row['pa_mpjpe_h0h1']:.1f}mm "
                      f"pcc@0.10={row.get('pcc@0.10', float('nan')):.3f}")

            # written after every run so a crash late in the sweep keeps results
            write_outputs(rows, cfg, cmd_args, dataset, n_used)

    write_curves(rows, cmd_args.output_folder)
    write_pcc(rows, cmd_args.output_folder)
    guru.info(f'Done. Results in {cmd_args.output_folder}')


def write_outputs(rows, cfg, cmd_args, dataset, n_used):
    fieldnames = list(rows[0].keys())
    for row in rows:  # later rows must not introduce new columns
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(osp.join(cmd_args.output_folder, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(osp.join(cmd_args.output_folder, 'metrics.json'), 'w') as f:
        json.dump({
            'rows': rows,
            'setup': {
                'dataset': cmd_args.dataset_name,
                'split': cmd_args.dataset_split,
                'n_samples_in_split': len(dataset),
                'n_samples_evaluated': n_used,
                'allow_missing_bev': cmd_args.allow_missing_bev,
                'batch_size': cfg.batch_size,
                'skip_steps': cmd_args.skip_steps,
                'seed': cmd_args.seed,
                'contact_threshold': CONTACT_THRESHOLD,
                'joints': cmd_args.joints,
                'guidance_params': list(cfg.model.regressor.experiment.guidance_params),
                'exp_cfg': cmd_args.exp_cfgs,
            },
        }, f, indent=2)


if __name__ == '__main__':
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
