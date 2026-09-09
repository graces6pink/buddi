import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from llib.data.preprocess.utils.check_interx_penetration import score_batch
from llib.methods.hhc_diffusion.evaluation.compare_utils import load_final_pkl, resolve_run_dir
from llib.methods.hhc_diffusion.evaluation.eval import fid_featurize, fid_on_params
from llib.utils.metrics.contact import MaxIntersection
from llib.utils.metrics.diffusion import GenDiversity, GenFID
from llib.utils.threed.distance import ContactMap

'''
Quantitative A-vs-B comparison for two sample.py --save-vis runs, computed
directly from their x_starts_smplx.pkl (no re-sampling, no real/ground-truth
data needed). See compare_samples_vis.py for the qualitative counterpart.

EvalModule.forward_generative_metrics (llib/methods/hhc_diffusion/eval_module.py)
is NOT used here: its gen_contact_and_isect branch is an unimplemented `pass`,
and its gen_fid branch needs a neural featurizer checkpoint
(essentials/buddi/fid_model.pt) that does not exist in this repo. Instead this
script directly reuses the lower-level pieces that eval.py and
check_interx_penetration.py already exercise successfully:
  - FID on raw stacked SMPL-X params (eval.py::fid_featurize/fid_on_params)
  - GenDiversity on vertices (llib/utils/metrics/diffusion.py)
  - MaxIntersection + check_interx_penetration.py::score_batch for
    penetration depth (avoids MaxIntersection.forward_batch's bare
    `except: ipdb.set_trace()`)
  - ContactMap.get_full_heatmap (llib/utils/threed/distance.py) thresholded
    by --contact-threshold for a no-ground-truth contact incidence proxy

Both runs here are UNCONDITIONAL, unseeded samples -- these metrics compare
the two batches' distributions/statistics against each other, not paired
per-example correspondences.

Usage:
  python llib/methods/hhc_diffusion/evaluation/compare_samples_metrics.py \
      --dir-a demo/diffusion/samples/sample_baseline_model \
      --dir-b demo/diffusion/samples/interx_uncond_1
'''

CONTACT_REGION_TO_VERTEX = 'essentials/contact/flickrci3ds_r75_rid_to_smplx_vid.pkl'
BODY_MODEL_UTILS_FOLDER = 'essentials/body_model_utils'


def to_device(params, device):
    return {k: v.to(device).float() for k, v in params.items()}


def compute_fid(params_a, params_b, device):
    feat_a = fid_featurize(to_device(params_a, device))
    feat_b = fid_featurize(to_device(params_b, device))
    return float(fid_on_params(feat_a, feat_b, GenFID()))


def diversity_within(vertices, device):
    diversity_metric = GenDiversity()
    v = vertices.to(device)
    n = v.shape[0] // 2
    return float(diversity_metric(v[:n], v[n:2 * n]))


def compute_diversity(vertices_a, vertices_b, device):
    diversity_metric = GenDiversity()
    va, vb = vertices_a.to(device), vertices_b.to(device)

    result = {
        'diversity_within_a': diversity_within(va, device),
        'diversity_within_b': diversity_within(vb, device),
    }
    n = min(va.shape[0], vb.shape[0])
    result['diversity_a_vs_b'] = float(diversity_metric(va[:n], vb[:n]))
    return result


def batched(n, batch_size):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def compute_penetration(vertices, device, batch_size, lowres):
    crit = MaxIntersection(
        model_type='smplx', body_model_utils_folder=BODY_MODEL_UTILS_FOLDER
    ).to(device)

    v1_all, v2_all = vertices[:, 0].to(device), vertices[:, 1].to(device)
    max_pen = []
    for s, e in batched(vertices.shape[0], batch_size):
        with torch.no_grad():
            scores = score_batch(crit, v1_all[s:e], v2_all[s:e], lowres)
        for i in range(e - s):
            max_pen.append(max(scores['max_v1_in_v2'][i], scores['max_v2_in_v1'][i]))

    max_pen = np.array(max_pen)
    has_pen = max_pen > 0.0
    pen_cm = max_pen * 100
    percentiles = [50, 75, 90, 95, 99, 100]
    result = {
        'penetration_rate': float(has_pen.mean()),
        'mean_max_penetration_cm': float(pen_cm.mean()),
        'mean_max_penetration_cm_among_penetrating': (
            float(pen_cm[has_pen].mean()) if has_pen.any() else 0.0
        ),
    }
    for p, v in zip(percentiles, np.percentile(pen_cm, percentiles)):
        result[f'penetration_p{p}_cm'] = float(v)
    return result


def compute_contact_rate(vertices, device, batch_size, contact_threshold):
    contact_map = ContactMap(region_to_vertex=CONTACT_REGION_TO_VERTEX).to(device)

    v1_all, v2_all = vertices[:, 0].to(device), vertices[:, 1].to(device)
    active_region_pairs = []
    for s, e in batched(vertices.shape[0], batch_size):
        with torch.no_grad():
            heatmap = contact_map.get_full_heatmap(v1_all[s:e], v2_all[s:e])
        contact_mask = heatmap < contact_threshold
        active_region_pairs.extend(contact_mask.sum(dim=(1, 2)).cpu().tolist())

    active_region_pairs = np.array(active_region_pairs)
    return {
        'contact_rate': float((active_region_pairs > 0).mean()),
        'mean_active_region_pairs': float(active_region_pairs.mean()),
    }


def print_table(label_a, label_b, metrics_a, metrics_b, fid_a_vs_b):
    print(f"\n=== A: {label_a}  vs  B: {label_b} ===")
    print(f"FID (A vs B, raw SMPL-X params): {fid_a_vs_b:.4f}")
    keys = list(metrics_a.keys())
    print(f"{'metric':<45}{'A':>15}{'B':>15}")
    for k in keys:
        print(f"{k:<45}{metrics_a[k]:>15.4f}{metrics_b[k]:>15.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-a', required=True)
    parser.add_argument('--dir-b', required=True)
    parser.add_argument('--label-a', default=None)
    parser.add_argument('--label-b', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lowres', type=int, default=500,
                         choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    parser.add_argument('--contact-threshold', type=float, default=0.02,
                         help='米。项目里没有现成的"接触判定距离阈值"常量'
                              '（真实接触标注来自人工标注，不是距离阈值），'
                              '这里 2cm 是一个启发式默认值，仅用于描述生成结果'
                              '本身的接触发生倾向，不代表和真值标注可比的指标')
    parser.add_argument('--out-json', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    label_a = args.label_a or osp.basename(osp.normpath(args.dir_a))
    label_b = args.label_b or osp.basename(osp.normpath(args.dir_b))
    out_json = args.out_json or osp.join(
        'demo/diffusion/samples/comparisons', f'{label_a}_vs_{label_b}', 'metrics.json'
    )

    run_a = resolve_run_dir(args.dir_a)
    run_b = resolve_run_dir(args.dir_b)
    params_a = load_final_pkl(run_a['x_starts_pkl'])
    params_b = load_final_pkl(run_b['x_starts_pkl'])

    fid_a_vs_b = compute_fid(params_a, params_b, device)

    metrics_a, metrics_b = {}, {}
    div = compute_diversity(params_a['vertices'], params_b['vertices'], device)
    metrics_a['diversity_within_set'] = div['diversity_within_a']
    metrics_b['diversity_within_set'] = div['diversity_within_b']

    pen_a = compute_penetration(params_a['vertices'], device, args.batch_size, args.lowres)
    pen_b = compute_penetration(params_b['vertices'], device, args.batch_size, args.lowres)
    metrics_a.update(pen_a)
    metrics_b.update(pen_b)

    contact_a = compute_contact_rate(params_a['vertices'], device, args.batch_size, args.contact_threshold)
    contact_b = compute_contact_rate(params_b['vertices'], device, args.batch_size, args.contact_threshold)
    metrics_a.update(contact_a)
    metrics_b.update(contact_b)

    print_table(label_a, label_b, metrics_a, metrics_b, fid_a_vs_b)
    print(f"diversity_a_vs_b (A、B 整体互相对照): {div['diversity_a_vs_b']:.4f}")
    print(
        '\n提示：两组都是无条件、无固定随机种子的采样，以上指标比较的是两批样本的'
        '整体分布/统计特性，不是逐样本的精确对照；接触发生率是基于距离阈值的启发式'
        '代理指标，不是和人工标注真值的一致性度量。'
    )

    if osp.dirname(out_json):
        os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({
            'label_a': label_a, 'label_b': label_b,
            'fid_a_vs_b': fid_a_vs_b,
            'diversity_a_vs_b': div['diversity_a_vs_b'],
            'metrics_a': metrics_a, 'metrics_b': metrics_b,
        }, f, indent=2)
    print(f"\n结果已写入 {out_json}")


if __name__ == '__main__':
    main()
