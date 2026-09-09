import os
import os.path as osp
import argparse
import pickle
import numpy as np
import torch
import smplx
from tqdm import tqdm
from loguru import logger as guru

from llib.utils.threed.intersection import winding_numbers
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.utils.metrics.contact import MaxIntersection

'''
Data-cleaning tool: scans the frames already sampled by process_interx.py
(dist_thresh=0.6, stride=20 -- the "large contact area" InterX subset) and
scores true body-mesh interpenetration between the two people, not just
root-translation proximity. A frame only counts as "penetrating" if at least
one SMPL-X vertex of one person is actually inside the other person's closed
mesh surface (generalized winding number >= 0.99), which is the same test
llib/utils/metrics/contact.py::MaxIntersection is built around.

We deliberately do NOT call MaxIntersection.forward()/forward_batch()
directly: forward_batch (llib/utils/metrics/contact.py:190) wraps its
per-frame distance computation in a bare `except: import ipdb;
ipdb.set_trace()`, which on an unattended scan over ~95k frames would hang
the whole job on stdin instead of crashing loudly the moment any frame hits
an edge case. Instead we reuse MaxIntersection's safe helper methods
(prep_mesh/close_mouth/to_lowres, which just build the watertight + lowres
mesh once) and reimplement the per-frame scoring loop ourselves with no
silent-hang failure mode. Verified to produce identical output to the
original forward_batch on real InterX batches before relying on it here.

Usage:
  python llib/data/preprocess/utils/check_interx_penetration.py \
      --processed-data-folder datasets/processed/InterX --limit-takes 50
'''


def load_flat_samples(processed_data_folder, limit_takes=-1, in_fn='processed.pkl'):
    """Scan the frame-level file, not the six-view expansion.

    Penetration is a property of the mocap frame -- the two bodies either
    interpenetrate or they do not, and no choice of camera changes that. Running
    this over processed_mv.pkl would do the identical SMPL-X forward six times
    per frame for identical answers, and would key the exclude list by the
    per-view imgname, which llib/data/preprocess/interx.py then has to undo.
    """
    processed_fn = osp.join(processed_data_folder, in_fn)
    processed = pickle.load(open(processed_fn, 'rb'))
    take_ids = list(processed.keys())
    if limit_takes > 0:
        take_ids = take_ids[:limit_takes]

    flat = []
    for take_id in take_ids:
        for sample_idx, sample in enumerate(processed[take_id]):
            flat.append((take_id, sample_idx, sample))
    return flat


def score_batch(crit, v1, v2, lowres):
    """Reimplementation of MaxIntersection.forward_batch without the bare
    `except: ipdb.set_trace()` landmine -- logic verified identical on real
    data. v1/v2: [B, V, 3] SMPL-X vertices for person0/person1."""
    out = {'min_dist': [], 'max_v1_in_v2': [], 'mean_v1_in_v2': [],
           'max_v2_in_v1': [], 'mean_v2_in_v1': []}

    _, t1l = crit.prep_mesh(v1, lowres)
    _, t2l = crit.prep_mesh(v2, lowres)
    interior_v1 = winding_numbers(v1, t2l).ge(0.99)
    interior_v2 = winding_numbers(v2, t1l).ge(0.99)

    for bidx in range(v1.shape[0]):
        v1v2 = pcl_pcl_pairwise_distance(v1[[bidx]], v2[[bidx]], squared=False)
        out['min_dist'].append(v1v2.min().item())

        max_val, mean_val = 0.0, 0.0
        if interior_v1[bidx].any():
            v1_to_v2 = v1v2[:, interior_v1[bidx], :].min(2)[0]
            max_val = v1_to_v2.max().item()
            mean_val = v1_to_v2.mean().item()
        out['max_v1_in_v2'].append(max_val)
        out['mean_v1_in_v2'].append(mean_val)

        max_val, mean_val = 0.0, 0.0
        if interior_v2[bidx].any():
            v2_to_v1 = v1v2[:, :, interior_v2[bidx]].min(1)[0]
            max_val = v2_to_v1.max().item()
            mean_val = v2_to_v1.mean().item()
        out['max_v2_in_v1'].append(max_val)
        out['mean_v2_in_v1'].append(mean_val)

    return out


def run_scan(flat_samples, batch_size, lowres, device):
    crit = MaxIntersection(
        model_type='smplx', body_model_utils_folder='essentials/body_model_utils'
    ).to(device)
    body_model = smplx.create(
        model_path='essentials/body_models', model_type='smplx',
        gender='neutral', num_betas=10, batch_size=batch_size * 2,
    ).to(device)

    results = []
    n_batches = (len(flat_samples) + batch_size - 1) // batch_size
    for bstart in tqdm(range(0, len(flat_samples), batch_size), total=n_batches):
        batch = flat_samples[bstart:bstart + batch_size]
        b_real = len(batch)
        # pad the last partial batch by repeating its last frame, so the
        # body model (fixed batch_size) always gets a full batch -- padded
        # results are computed but simply dropped below.
        padded = batch + [batch[-1]] * (batch_size - b_real)

        go = np.stack([s['pgt_smplx_global_orient'] for _, _, s in padded])   # [B,2,3]
        bp = np.stack([s['pgt_smplx_body_pose'] for _, _, s in padded])       # [B,2,63]
        betas = np.stack([s['pgt_smplx_betas'] for _, _, s in padded])        # [B,2,10]
        transl = np.stack([s['pgt_smplx_transl'] for _, _, s in padded])      # [B,2,3]

        go_t = torch.from_numpy(go).float().to(device).reshape(batch_size * 2, 3)
        bp_t = torch.from_numpy(bp).float().to(device).reshape(batch_size * 2, 63)
        betas_t = torch.from_numpy(betas).float().to(device).reshape(batch_size * 2, 10)
        transl_t = torch.from_numpy(transl).float().to(device).reshape(batch_size * 2, 3)

        with torch.no_grad():
            out = body_model(global_orient=go_t, body_pose=bp_t, betas=betas_t, transl=transl_t)
            verts = out.vertices.reshape(batch_size, 2, -1, 3)
            v1, v2 = verts[:, 0].contiguous(), verts[:, 1].contiguous()
            scores = score_batch(crit, v1, v2, lowres)

        for i in range(b_real):
            take_id, sample_idx, sample = batch[i]
            max_pen = max(scores['max_v1_in_v2'][i], scores['max_v2_in_v1'][i])
            results.append({
                'take_id': take_id,
                'sample_idx': sample_idx,
                'imgname': sample['imgname'],
                'min_dist': scores['min_dist'][i],
                'max_v1_in_v2': scores['max_v1_in_v2'][i],
                'mean_v1_in_v2': scores['mean_v1_in_v2'][i],
                'max_v2_in_v1': scores['max_v2_in_v1'][i],
                'mean_v2_in_v1': scores['mean_v2_in_v1'][i],
                'max_penetration': max_pen,
                'has_penetration': max_pen > 0.0,
            })

    return results


def print_summary(results):
    max_pen = np.array([r['max_penetration'] for r in results])
    has_pen = np.array([r['has_penetration'] for r in results])

    print(f'\n=== 穿模检测结果（共 {len(results)} 帧）===')
    print(f'检测到身体网格穿插的帧数: {has_pen.sum()} ({has_pen.mean()*100:.1f}%)')

    print('\n=== 穿透深度分布（含未穿模的0值）===')
    percentiles = [50, 75, 90, 95, 99, 99.5, 100]
    for p, v in zip(percentiles, np.percentile(max_pen, percentiles)):
        print(f'  p{p:>5}: {v*100:.2f} cm')

    if has_pen.any():
        print('\n=== 穿透深度分布（仅统计有穿模的帧）===')
        pen_only = max_pen[has_pen]
        for p, v in zip(percentiles, np.percentile(pen_only, percentiles)):
            print(f'  p{p:>5}: {v*100:.2f} cm')

    print('\n=== 不同阈值下会排除多少帧 ===')
    for thresh_cm in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        frac = (max_pen > thresh_cm / 100.0).mean()
        print(f'  排除阈值 {thresh_cm:>4.1f}cm: 会排除 {frac*100:.1f}% 的帧')


def maybe_plot(results, out_folder):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        guru.warning('matplotlib not available, skipping plot')
        return

    os.makedirs(out_folder, exist_ok=True)
    max_pen_cm = np.array([r['max_penetration'] for r in results]) * 100
    pen_only = max_pen_cm[max_pen_cm > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(max_pen_cm, bins=100)
    axes[0].set_title('max penetration depth per frame (all frames)')
    axes[0].set_xlabel('cm')

    if len(pen_only) > 0:
        axes[1].hist(pen_only, bins=100)
    axes[1].set_title('max penetration depth (only frames with penetration)')
    axes[1].set_xlabel('cm')

    fig.tight_layout()
    out_fn = osp.join(out_folder, 'penetration_depth_hist.png')
    fig.savefig(out_fn, dpi=130)
    guru.info(f'Saved plot to {out_fn}')


def save_exclude_list(results, threshold_m, out_fn):
    """Saved as a flat set of `imgname` strings (unique across the whole
    dataset, see process_interx.py) rather than {take_id: [sample_idx,...]}
    -- imgname survives cache flattening/reloading in interx.py::load(),
    where per-take list positions would not."""
    exclude = {r['imgname'] for r in results if r['max_penetration'] > threshold_m}
    with open(out_fn, 'wb') as f:
        pickle.dump(exclude, f)
    guru.info(f'threshold={threshold_m*100:.1f}cm: {len(exclude)}/{len(results)} 帧会被排除, '
               f'已保存到 {out_fn}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    parser.add_argument('--in-fn', default='processed.pkl',
                        help='frame-level file to scan. Leave at the default: the '
                             'six-view processed_mv.pkl would just repeat every frame '
                             'six times for the same answer.')
    parser.add_argument('--batch-size', type=int, default=16, help='frames per batch (each frame = 2 people)')
    parser.add_argument('--lowres', type=int, default=500, choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                         help='target-mesh resolution for winding-number test; 500 is the accuracy/speed sweet spot (verified: 500≈1000, 100/300 miss shallow penetration)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit-takes', type=int, default=-1, help='only scan the first N takes, for a quick dry run')
    parser.add_argument('--exclude-threshold', type=float, default=None,
                         help='meters; if set, also write a penetration_exclude_list.pkl (a set of imgname strings) for frames whose max_penetration exceeds this')
    parser.add_argument('--force-rescan', action='store_true', default=False,
                         help='rerun the GPU scan even if a cached penetration_report.pkl already exists (by default we reuse it, since picking a different --exclude-threshold does not require rescanning)')
    args = parser.parse_args()

    diag_folder = osp.join(args.processed_data_folder, 'diagnostics')
    os.makedirs(diag_folder, exist_ok=True)
    report_fn = osp.join(diag_folder, 'penetration_report.pkl')

    if osp.exists(report_fn) and not args.force_rescan and args.limit_takes <= 0:
        guru.info(f'Found cached report at {report_fn}, reusing it (pass --force-rescan to redo the GPU scan)')
        with open(report_fn, 'rb') as f:
            results = pickle.load(f)
    else:
        flat_samples = load_flat_samples(args.processed_data_folder,
                                         limit_takes=args.limit_takes, in_fn=args.in_fn)
        guru.info(f'Loaded {len(flat_samples)} frames to scan (limit_takes={args.limit_takes})')

        results = run_scan(flat_samples, args.batch_size, args.lowres, args.device)

        with open(report_fn, 'wb') as f:
            pickle.dump(results, f)
        guru.info(f'Saved per-frame report to {report_fn}')

    print_summary(results)
    maybe_plot(results, diag_folder)

    if args.exclude_threshold is not None:
        save_exclude_list(results, args.exclude_threshold, osp.join(diag_folder, 'penetration_exclude_list.pkl'))


if __name__ == '__main__':
    main()
