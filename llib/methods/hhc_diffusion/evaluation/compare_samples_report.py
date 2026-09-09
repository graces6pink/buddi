import argparse
import itertools
import json
import os
import os.path as osp

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from llib.methods.hhc_diffusion.evaluation.compare_samples_metrics import (
    compute_contact_rate, compute_fid, compute_penetration, diversity_within,
)
from llib.methods.hhc_diffusion.evaluation.compare_utils import load_final_pkl, resolve_run_dir

'''
N-way version of compare_samples_metrics.py: same underlying metric functions
(FID / diversity / penetration incl. percentiles / contact rate), but for an
arbitrary list of runs instead of exactly two -- built for comparing
sample_baseline_model, interx_uncond_1 and interx_uncond_clean_1 (all now
1000 samples each) side by side in one report.

Per-run stats (diversity/penetration/contact) are computed once per run; FID
is pairwise across all run combinations (n choose 2), since it's inherently
a distance between two distributions.

Also writes a small overview grid image (one row per run, a handful of evenly
spaced samples, single frame each) -- a light qualitative sanity check, not a
full per-sample GIF comparison (see compare_samples_vis.py for that, pairwise
only).

Usage:
  python llib/methods/hhc_diffusion/evaluation/compare_samples_report.py \
      --dirs demo/diffusion/samples/sample_baseline_model \
             demo/diffusion/samples/interx_uncond_1 \
             demo/diffusion/samples/interx_uncond_clean_1 \
      --labels baseline interx_uncond interx_uncond_clean
'''


def build_overview_grid(runs, out_path, cols=12, thumb_size=(110, 140), frame_index=None):
    tw, th = thumb_size
    row_h = th + 16
    sheet = Image.new('RGB', (cols * tw, len(runs) * row_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for r, run in enumerate(runs):
        gifs = sorted(os.listdir(run['renders_dir']))
        idxs = np.linspace(0, len(gifs) - 1, cols).astype(int)
        draw.text((4, r * row_h + 2), run['label'], fill=(255, 255, 255))
        for c, gidx in enumerate(idxs):
            frames = imageio.mimread(osp.join(run['renders_dir'], gifs[gidx]))
            fi = frame_index if frame_index is not None else len(frames) // 2
            fi = min(fi, len(frames) - 1)
            thumb = Image.fromarray(np.array(frames[fi])[..., :3]).resize((tw, th))
            sheet.paste(thumb, (c * tw, r * row_h + 16))

    sheet.save(out_path)
    print(f'[compare_samples_report] wrote overview grid to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lowres', type=int, default=500,
                         choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    parser.add_argument('--contact-threshold', type=float, default=0.02)
    parser.add_argument('--overview-cols', type=int, default=12)
    parser.add_argument('--skip-overview', action='store_true')
    args = parser.parse_args()

    labels = args.labels or [osp.basename(osp.normpath(d)) for d in args.dirs]
    assert len(labels) == len(args.dirs), '--labels must match --dirs in length'
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = args.out_dir or osp.join('demo/diffusion/samples/comparisons', '_'.join(labels))
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for label, d in zip(labels, args.dirs):
        resolved = resolve_run_dir(d)
        resolved['label'] = label
        resolved['params'] = load_final_pkl(resolved['x_starts_pkl'])
        runs.append(resolved)
        print(f'[compare_samples_report] {label}: {resolved["params"]["vertices"].shape[0]} samples '
              f'from {resolved["run_dir"]}')

    print('\n[compare_samples_report] computing per-run stats (diversity / penetration / contact)...')
    per_run_metrics = {}
    for run in runs:
        verts = run['params']['vertices']
        m = {'diversity_within_set': diversity_within(verts, device)}
        m.update(compute_penetration(verts, device, args.batch_size, args.lowres))
        m.update(compute_contact_rate(verts, device, args.batch_size, args.contact_threshold))
        per_run_metrics[run['label']] = m
        print(f'  done: {run["label"]}')

    print('\n[compare_samples_report] computing pairwise FID...')
    fid_matrix = {}
    for run_i, run_j in itertools.combinations(runs, 2):
        fid = compute_fid(run_i['params'], run_j['params'], device)
        fid_matrix[f'{run_i["label"]}__vs__{run_j["label"]}'] = fid
        print(f'  FID({run_i["label"]}, {run_j["label"]}) = {fid:.4f}')

    print('\n=== per-run metrics ===')
    keys = list(next(iter(per_run_metrics.values())).keys())
    metric_header = 'metric'
    header = f"{metric_header:<40}" + "".join(f"{l:>20}" for l in labels)
    print(header)
    for k in keys:
        print(f"{k:<40}" + "".join(f"{per_run_metrics[l][k]:>20.4f}" for l in labels))

    result = {
        'labels': labels,
        'dirs': args.dirs,
        'n_samples': {r['label']: int(r['params']['vertices'].shape[0]) for r in runs},
        'per_run_metrics': per_run_metrics,
        'fid_pairwise': fid_matrix,
        'contact_threshold_m': args.contact_threshold,
        'lowres': args.lowres,
    }
    out_json = osp.join(out_dir, 'report.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n[compare_samples_report] wrote {out_json}')

    if not args.skip_overview:
        build_overview_grid(runs, osp.join(out_dir, 'overview_grid.png'), cols=args.overview_cols)

    print(
        '\nNote: all runs are unconditional, unseeded samples -- these metrics compare '
        'each run\'s overall distribution/statistics, not paired per-example correspondences; '
        'contact rate is a heuristic distance-threshold proxy, not agreement with human-annotated '
        'ground truth.'
    )


if __name__ == '__main__':
    main()
