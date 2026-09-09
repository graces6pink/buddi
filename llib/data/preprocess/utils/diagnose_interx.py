import os.path as osp
import argparse
import random
import numpy as np
from tqdm import tqdm
from loguru import logger as guru

from llib.data.preprocess.utils.process_interx import find_takes, load_person, detect_interaction_segments

'''
Diagnostic tool to help pick --dist-thresh / --stride for process_interx.py
BEFORE committing to a full preprocessing run. Samples a subset of takes,
reports the distribution of the inter-person root-translation distance, and
for a grid of (dist_thresh, stride) combos reports how many training frames
/ segments / takes would actually survive -- extrapolated to the full
dataset size so you can pick a target training-set size up front.

Run from repo root, e.g.:
  python llib/data/preprocess/utils/diagnose_interx.py --sample-takes 1000
'''


def collect_distances(motions_folder, sample_takes=1000, seed=0):
    all_takes = find_takes(motions_folder)
    guru.info(f'Found {len(all_takes)} takes total under {motions_folder}')

    rng = random.Random(seed)
    if sample_takes > 0 and sample_takes < len(all_takes):
        takes = rng.sample(all_takes, sample_takes)
    else:
        takes = all_takes
    guru.info(f'Sampling {len(takes)} takes for diagnostics')

    per_take = {}  # take_id -> filtered dist array (post medfilt, same as process_interx.py)
    failed = 0
    for take_folder in tqdm(takes):
        take_id = osp.basename(take_folder.rstrip('/'))
        try:
            p1 = load_person(osp.join(take_folder, 'P1.npz'))
            p2 = load_person(osp.join(take_folder, 'P2.npz'))
        except Exception as e:
            failed += 1
            continue
        # reuse the exact same distance signal the real pipeline uses
        _, dist = detect_interaction_segments(p1['transl'], p2['transl'])
        per_take[take_id] = dist

    if failed:
        guru.warning(f'{failed} takes failed to load and were skipped')

    return per_take, len(all_takes)


def print_distance_stats(per_take):
    all_dist = np.concatenate(list(per_take.values()))
    take_mins = np.array([d.min() for d in per_take.values()])
    take_lens = np.array([len(d) for d in per_take.values()])

    print('\n=== 逐帧距离分布（medfilt 之后，跟 process_interx.py 用的是同一个信号）===')
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(all_dist, percentiles)
    for p, v in zip(percentiles, vals):
        print(f'  p{p:>2d}: {v:.3f} m')
    print(f'  min={all_dist.min():.3f}  max={all_dist.max():.3f}  mean={all_dist.mean():.3f}')

    print('\n=== 每个 take 的「最近距离」分布（判断有没有 take 从头到尾都没靠近过）===')
    for p, v in zip(percentiles, np.percentile(take_mins, percentiles)):
        print(f'  p{p:>2d}: {v:.3f} m')
    for thresh in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
        frac = (take_mins > thresh).mean()
        print(f'  最近距离 > {thresh:.1f}m 的 take 占比: {frac*100:.1f}%  (=在该阈值下这些take会被完全排除)')

    print(f'\n=== take 长度（帧数）分布 ===')
    for p, v in zip(percentiles, np.percentile(take_lens, percentiles)):
        print(f'  p{p:>2d}: {v:.0f} frames')

    return all_dist, take_mins, take_lens


def print_per_take_coverage(per_take, dist_threshs):
    """
    For each threshold: within a single take, what FRACTION of its frames
    are close enough to count as "large contact area"? (raw is_close mean,
    not the padded/min-len-filtered segment logic used for actual sampling)
    This is a different question from "how many takes get excluded" --
    it tells you, among the takes that DO qualify, how much usable signal
    each one actually has.
    """
    percentiles = [10, 25, 50, 75, 90]
    print('\n=== 每个 take 内部「有效帧占比」分布（不同阈值下）===')
    print('(只统计有至少1帧满足阈值的take；0帧的take在上面的"完全排除"统计里已经算过)')
    header = f'{"dist_thresh":>11} {"n_valid_takes":>14} ' + ' '.join(f'p{p:>2d}' for p in percentiles)
    print(header)
    for thresh in dist_threshs:
        fracs = []
        for dist in per_take.values():
            is_close = dist < thresh
            if is_close.any():
                fracs.append(is_close.mean())
        fracs = np.array(fracs)
        vals = np.percentile(fracs, percentiles) if len(fracs) else np.zeros(len(percentiles))
        row = f'{thresh:>11.2f} {len(fracs):>14d} ' + ' '.join(f'{v*100:>4.0f}%' for v in vals)
        print(row)


def grid_search(per_take, total_takes_in_dataset, dist_threshs, strides, min_len=15, pad=8):
    n_sampled = len(per_take)
    scale = total_takes_in_dataset / max(n_sampled, 1)

    print(f'\n=== (dist_thresh, stride) 网格搜索（在 {n_sampled} 个采样 take 上算，'
          f'再按 {scale:.2f}x 缩放到全量 {total_takes_in_dataset} take 做估计）===')
    header = f'{"dist_thresh":>11} {"stride":>7} {"segments":>10} {"empty_takes":>12} {"samples(采样)":>14} {"samples(估计全量)":>18}'
    print(header)
    print('-' * len(header))

    results = []
    for dist_thresh in dist_threshs:
        # segments only depend on dist_thresh/min_len/pad, not on stride -- recompute once per thresh
        take_segments = {}
        for take_id, dist in per_take.items():
            is_close = dist < dist_thresh
            raw = []
            start = None
            for t, close in enumerate(is_close):
                if close and start is None:
                    start = t
                elif not close and start is not None:
                    raw.append((start, t))
                    start = None
            if start is not None:
                raw.append((start, len(is_close)))
            segs = []
            n = len(dist)
            for s, e in raw:
                if e - s < min_len:
                    continue
                segs.append((max(0, s - pad), min(n, e + pad)))
            take_segments[take_id] = segs

        total_segments = sum(len(v) for v in take_segments.values())
        empty_takes = sum(1 for v in take_segments.values() if len(v) == 0)

        for stride in strides:
            total_samples = 0
            for segs in take_segments.values():
                for s, e in segs:
                    total_samples += len(range(s, e, stride))
            results.append({
                'dist_thresh': dist_thresh, 'stride': stride,
                'segments': total_segments, 'empty_takes': empty_takes,
                'samples_sampled': total_samples,
                'samples_full_estimate': int(total_samples * scale),
            })
            print(f'{dist_thresh:>11.2f} {stride:>7d} {total_segments:>10d} {empty_takes:>12d} '
                  f'{total_samples:>14d} {int(total_samples*scale):>18d}')
    return results


def maybe_plot(all_dist, take_mins, out_folder):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        guru.warning('matplotlib not available, skipping plots')
        return

    import os
    os.makedirs(out_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(all_dist, bins=100)
    axes[0].set_title('per-frame P1-P2 root translation distance')
    axes[0].set_xlabel('meters')

    axes[1].hist(take_mins, bins=100)
    axes[1].set_title('per-take minimum distance')
    axes[1].set_xlabel('meters')

    fig.tight_layout()
    out_fn = osp.join(out_folder, 'interx_distance_diagnostics.png')
    fig.savefig(out_fn, dpi=130)
    guru.info(f'Saved plot to {out_fn}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-data-folder', default='datasets/original/InterX')
    parser.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    parser.add_argument('--sample-takes', type=int, default=1000, help='randomly sample this many takes for speed; <=0 means use all takes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dist-threshs', type=float, nargs='+', default=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 10, 20, 30])
    args = parser.parse_args()

    motions_folder = osp.join(args.original_data_folder, 'motions')
    per_take, total_takes = collect_distances(motions_folder, sample_takes=args.sample_takes, seed=args.seed)

    all_dist, take_mins, take_lens = print_distance_stats(per_take)
    print_per_take_coverage(per_take, args.dist_threshs)
    grid_search(per_take, total_takes, args.dist_threshs, args.strides)
    maybe_plot(all_dist, take_mins, osp.join(args.processed_data_folder, 'diagnostics'))


if __name__ == '__main__':
    main()
