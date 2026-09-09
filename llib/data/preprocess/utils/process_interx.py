import os
import os.path as osp
import argparse
import pickle
import numpy as np
from collections import defaultdict

from scipy.signal import medfilt
from tqdm import tqdm
from loguru import logger as guru

'''
Offline preprocessing script for a custom Inter-X-format dataset: two-person
motion sequences stored as P1.npz / P2.npz per take, each with SMPL-X ground
truth (pose_body, pose_lhand, pose_rhand, betas, root_orient, trans, gender).

A take may mix non-interacting and interacting parts, so this script auto-
detects the interacting sub-segments (via root-translation proximity between
the two people), subsamples frames within those segments, and writes out a
processed.pkl + train_val_split.npz that llib/data/preprocess/interx.py reads.

Only global_orient/body_pose/betas/transl are used downstream (see
llib/methods/hhc_diffusion/loss_module.py) -- hand pose and gender are read
but not stored, and bev_* fields are filled with zeros since this dataset is
only used to train the unconditional BUDDI variant for now (guidance_params
is empty, so bev_* is never consumed, see llib/methods/hhc_diffusion/
train_module.py::get_guidance_params).
'''


def find_takes(motions_folder):
    """Each leaf folder containing P1.npz and P2.npz is one take."""
    takes = []
    for root, _dirs, files in os.walk(motions_folder):
        if 'P1.npz' in files and 'P2.npz' in files:
            takes.append(root)
    return sorted(takes)


def load_person(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    num_frames = d['pose_body'].shape[0]
    return {
        'global_orient': d['root_orient'].astype(np.float32),  # [T,3]
        'body_pose': d['pose_body'].reshape(num_frames, -1).astype(np.float32),  # [T,63]
        'betas': np.tile(d['betas'][0:1].astype(np.float32), (num_frames, 1)),  # [T,10]
        'transl': d['trans'].astype(np.float32),  # [T,3]
    }


def detect_interaction_segments(
    transl1, transl2, dist_thresh=0.6, min_len=15, pad=8, medfilt_kernel=7
):
    """
    Heuristic: two people are "interacting" when their root translations
    (proxy for pelvis position) are within dist_thresh of each other.
    Returns a list of (start, end) frame index tuples (end exclusive).
    """
    dist = np.linalg.norm(transl1 - transl2, axis=1)
    if len(dist) >= medfilt_kernel:
        dist = medfilt(dist, kernel_size=medfilt_kernel)
    is_close = dist < dist_thresh

    raw_segments = []
    start = None
    for t, close in enumerate(is_close):
        if close and start is None:
            start = t
        elif not close and start is not None:
            raw_segments.append((start, t))
            start = None
    if start is not None:
        raw_segments.append((start, len(is_close)))

    num_frames = len(dist)
    segments = []
    for s, e in raw_segments:
        if e - s < min_len:
            continue
        s_pad = max(0, s - pad)
        e_pad = min(num_frames, e + pad)
        segments.append((s_pad, e_pad))
    return segments, dist


def process_take(take_folder, stride=20, dist_thresh=0.6, min_len=15, pad=8):
    p1 = load_person(osp.join(take_folder, 'P1.npz'))
    p2 = load_person(osp.join(take_folder, 'P2.npz'))

    segments, dist = detect_interaction_segments(
        p1['transl'], p2['transl'],
        dist_thresh=dist_thresh, min_len=min_len, pad=pad,
    )

    take_id = osp.basename(take_folder.rstrip('/'))
    samples = []
    for seg_idx, (s, e) in enumerate(segments):
        for frame_idx in range(s, e, stride):
            imgname = f'{take_id}_{seg_idx}_{frame_idx}'
            sample = {
                'imgname': imgname,
                # 'InterX' substring is required -- llib/data/single.py::get_single_item
                # dispatches on a substring match in imgpath, see plan doc.
                'imgpath': f'InterX/{imgname}.png',
                'img_height': 900,
                'img_width': 900,
                'pgt_smplx_global_orient': np.stack(
                    [p1['global_orient'][frame_idx], p2['global_orient'][frame_idx]], axis=0),
                'pgt_smplx_body_pose': np.stack(
                    [p1['body_pose'][frame_idx], p2['body_pose'][frame_idx]], axis=0),
                'pgt_smplx_betas': np.stack(
                    [p1['betas'][frame_idx], p2['betas'][frame_idx]], axis=0),
                'pgt_smplx_transl': np.stack(
                    [p1['transl'][frame_idx], p2['transl'][frame_idx]], axis=0),
                'pgt_smplx_scale': np.zeros((2, 1), dtype=np.float32),
                # bev_* is unused while training unconditionally (guidance_params=[]),
                # kept as zeros only so llib/data/single.py::get_single_item doesn't KeyError.
                'bev_smplx_global_orient': np.zeros((2, 3), dtype=np.float32),
                'bev_smplx_body_pose': np.zeros((2, 63), dtype=np.float32),
                'bev_smplx_betas': np.zeros((2, 10), dtype=np.float32),
                'bev_smplx_transl': np.zeros((2, 3), dtype=np.float32),
                'bev_smplx_scale': np.zeros((2,), dtype=np.float32),
                'information_missing': False,
                'take_id': take_id,
            }
            samples.append(sample)
    return samples, dist, segments


def actor_group(take_id):
    """Inter-X take ids look like G041T009A034R034: G = performer pair, A = action,
    T/R = take and repetition indices. The leading G field is what has to stay
    inside one split."""
    return str(take_id)[:4]


def make_split(frames_per_take, ratios=(0.90, 0.05, 0.05), seed=0):
    """Split by performer group, balanced on frame count.

    Splitting take-by-take (what this used to do) leaks badly: measured on the
    released split, all 59 performer groups appeared in both train and val, and
    227 of 237 val takes had a sibling take in train with the same performer
    pair AND the same action. Validation was then scoring "same actors, same
    action, different repetition", which is memorisation plus interpolation, not
    generalisation -- and it is why the released run's 90mm looked so much better
    than the downstream FlickrCI3D numbers.

    Balancing on frames rather than on group count matters: the 59 groups differ
    by 6.1x in size (524 to 3189 frames, median 1506), so picking "3 groups for
    val" would land anywhere between 2% and 10% of the data. Greedy assignment to
    whichever bucket has the largest *relative* shortfall keeps the split close
    to the requested ratios whatever the group sizes are.
    """
    groups = defaultdict(list)
    for take_id in frames_per_take:
        groups[actor_group(take_id)].append(take_id)

    group_ids = sorted(groups)
    np.random.RandomState(seed).shuffle(group_ids)

    total = sum(frames_per_take.values())
    targets = [max(total * r, 1e-9) for r in ratios]
    buckets, filled = [[] for _ in ratios], [0.0] * len(ratios)

    for g in group_ids:
        n = sum(frames_per_take[t] for t in groups[g])
        i = int(np.argmax([(targets[k] - filled[k]) / targets[k] for k in range(len(ratios))]))
        buckets[i].extend(sorted(groups[g]))
        filled[i] += n

    for name, ids, n_frames, target in zip(('train', 'val', 'test'), buckets, filled, targets):
        n_groups = len({actor_group(t) for t in ids})
        guru.info(f'  {name}: {n_groups} groups, {len(ids)} takes, {int(n_frames)} frames '
                  f'({n_frames / total * 100:.1f}%, target {target / total * 100:.0f}%)')

    seen = [set(map(actor_group, b)) for b in buckets]
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            assert not (seen[a] & seen[b]), f'performer group in two splits: {seen[a] & seen[b]}'

    return [np.array(sorted(b)) for b in buckets]


def write_split(processed, processed_data_folder, seed=0):
    frames_per_take = {t: len(v) for t, v in processed.items()}
    train_ids, val_ids, test_ids = make_split(frames_per_take, seed=seed)
    split_fn = osp.join(processed_data_folder, 'train_val_split.npz')
    np.savez(split_fn, train=train_ids, val=val_ids, test=test_ids)
    guru.info(f'Saved split -> {split_fn}')
    guru.warning('Delete datasets/processed/InterX/{train,val,test}_diffusion.pkl now: '
                 'those are caches keyed by nothing and will not notice the split changed.')
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-data-folder', default='datasets/original/InterX')
    parser.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    parser.add_argument('--stride', type=int, default=40, help='frame stride within a detected interaction segment. Inter-X is 120fps, so 40 samples 3 frames per second. The released processed.pkl used 20 (6fps), which is largely temporal redundancy -- with six camera views per frame the budget is better spent on viewpoints than on near-duplicate frames.')
    parser.add_argument('--dist-thresh', type=float, default=0.6, help='root-translation distance (meters) below which two people count as interacting. 0.6 is the value used for the released processed.pkl -- it keeps close-contact actions (hug, support) and drops distant ones (wave, point): 4735/8026 takes, 95448 samples. A looser 1.2 keeps 7929 takes / 249300 samples.')
    parser.add_argument('--min-len', type=int, default=15, help='minimum segment length in frames, shorter candidates are dropped')
    parser.add_argument('--pad', type=int, default=8, help='frames of padding added before/after a detected segment')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1, help='only process the first N takes, useful for a quick dry run')
    parser.add_argument('--split-only', action='store_true',
                        help='do not touch processed.pkl; just recompute train_val_split.npz from the one already on disk. Changing the split needs no BEV re-run -- processed.pkl is keyed by take and the split file is only a partition table.')
    args = parser.parse_args()

    if args.split_only:
        processed_fn = osp.join(args.processed_data_folder, 'processed.pkl')
        processed = pickle.load(open(processed_fn, 'rb'))
        guru.info(f'Loaded {processed_fn}: {len(processed)} takes')
        write_split(processed, args.processed_data_folder, seed=args.seed)
        return

    motions_folder = osp.join(args.original_data_folder, 'motions')
    takes = find_takes(motions_folder)
    if args.limit > 0:
        takes = takes[:args.limit]
    guru.info(f'Found {len(takes)} takes under {motions_folder}')

    os.makedirs(args.processed_data_folder, exist_ok=True)

    processed = {}
    total_samples = 0
    total_segments = 0
    for take_folder in tqdm(takes):
        take_id = osp.basename(take_folder.rstrip('/'))
        try:
            samples, _dist, segments = process_take(
                take_folder, stride=args.stride, dist_thresh=args.dist_thresh,
                min_len=args.min_len, pad=args.pad,
            )
        except Exception as e:
            guru.warning(f'Failed to process {take_folder}: {e}')
            continue
        if len(samples) == 0:
            continue
        processed[take_id] = samples
        total_samples += len(samples)
        total_segments += len(segments)

    guru.info(
        f'Processed {len(processed)}/{len(takes)} takes with usable data, '
        f'{total_segments} interaction segments detected, {total_samples} total frame samples'
    )

    processed_fn = osp.join(args.processed_data_folder, 'processed.pkl')
    with open(processed_fn, 'wb') as f:
        pickle.dump(processed, f)
    guru.info(f'Saved {processed_fn}')

    write_split(processed, args.processed_data_folder, seed=args.seed)


if __name__ == '__main__':
    main()
