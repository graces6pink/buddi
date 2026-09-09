import argparse
import glob
import math
import os
import os.path as osp
import re

import imageio
import numpy as np
from PIL import Image, ImageDraw

from llib.methods.hhc_diffusion.evaluation.compare_utils import resolve_run_dir

'''
Batch-compares the turntable GIFs saved by sample.py --save-vis for two
sampling runs (see compare_utils.py for how the run folder is located).

Usage:
  python llib/methods/hhc_diffusion/evaluation/compare_samples_vis.py \
      --dir-a demo/diffusion/samples/sample_baseline_model \
      --dir-b demo/diffusion/samples/interx_uncond_1

Both runs here are UNCONDITIONAL and unseeded samples, so sample i in A and
sample i in B are two independent draws, not the same underlying pose -- the
outputs below give a batch-level qualitative impression, not a paired
per-example diff.
'''


def paired_indices(renders_a, renders_b):
    pattern = re.compile(r'(\d+)_gen\.gif$')

    def indices(d):
        out = {}
        for fn in glob.glob(osp.join(d, '*_gen.gif')):
            m = pattern.search(fn)
            if m:
                out[int(m.group(1))] = fn
        return out

    idx_a, idx_b = indices(renders_a), indices(renders_b)
    common = sorted(set(idx_a) & set(idx_b))
    if len(idx_a) != len(idx_b) or len(common) != len(idx_a):
        print(
            f'[compare_samples_vis] A 有 {len(idx_a)} 个样本，B 有 {len(idx_b)} 个样本，'
            f'取交集共 {len(common)} 个配对'
        )
    return [(i, idx_a[i], idx_b[i]) for i in common]


def label_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))
    return np.array(img)


def make_paired_gif(path_a, path_b, label_a, label_b, out_path, fps):
    frames_a = imageio.mimread(path_a)
    frames_b = imageio.mimread(path_b)
    n = min(len(frames_a), len(frames_b))

    combined = []
    for t in range(n):
        fa = label_frame(np.array(frames_a[t])[..., :3], label_a)
        fb = label_frame(np.array(frames_b[t])[..., :3], label_b)
        h = max(fa.shape[0], fb.shape[0])
        canvas = Image.new('RGB', (fa.shape[1] + fb.shape[1] + 2, h), (30, 30, 30))
        canvas.paste(Image.fromarray(fa), (0, 0))
        canvas.paste(Image.fromarray(fb), (fa.shape[1] + 2, 0))
        combined.append(np.array(canvas))

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, combined, fps=fps)


def build_contact_sheet(pairs, label_a, label_b, out_dir, frame_index, thumb_size,
                         cols, samples_per_page):
    tw, th = thumb_size
    cell_w, cell_h = tw, th * 2 + 12

    n_pages = math.ceil(len(pairs) / samples_per_page)
    for page in range(n_pages):
        page_pairs = pairs[page * samples_per_page:(page + 1) * samples_per_page]
        rows = math.ceil(len(page_pairs) / cols)
        header_h = 16
        sheet = Image.new('RGB', (cols * cell_w, header_h + rows * cell_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        draw.text((2, 2), f'top row = {label_a}   /   bottom row = {label_b}', fill=(255, 255, 255))

        for k, (idx, path_a, path_b) in enumerate(page_pairs):
            r, c = divmod(k, cols)
            x0, y0 = c * cell_w, header_h + r * cell_h

            frames_a = imageio.mimread(path_a)
            frames_b = imageio.mimread(path_b)
            fi_a = min(frame_index, len(frames_a) - 1)
            fi_b = min(frame_index, len(frames_b) - 1)
            thumb_a = Image.fromarray(np.array(frames_a[fi_a])[..., :3]).resize((tw, th))
            thumb_b = Image.fromarray(np.array(frames_b[fi_b])[..., :3]).resize((tw, th))

            sheet.paste(thumb_a, (x0, y0))
            sheet.paste(thumb_b, (x0, y0 + th))
            draw.text((x0 + 2, y0 + 1), f'{idx:05d}', fill=(255, 255, 0))
            draw.text((x0 + 2, y0 + th + 1), f'{idx:05d}', fill=(0, 255, 255))

        out_path = osp.join(out_dir, f'contact_sheet_{page:02d}.png')
        sheet.save(out_path)
        print(f'[compare_samples_vis] 已保存总览图 {out_path} ({len(page_pairs)} 个样本)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-a', required=True)
    parser.add_argument('--dir-b', required=True)
    parser.add_argument('--label-a', default=None)
    parser.add_argument('--label-b', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--frame-index', type=int, default=None,
                         help='总览图取哪一帧，默认取中间帧')
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--thumb-size', type=int, nargs=2, default=[100, 128])
    parser.add_argument('--samples-per-page', type=int, default=60)
    parser.add_argument('--skip-paired-gifs', action='store_true')
    parser.add_argument('--skip-contact-sheet', action='store_true')
    args = parser.parse_args()

    label_a = args.label_a or osp.basename(osp.normpath(args.dir_a))
    label_b = args.label_b or osp.basename(osp.normpath(args.dir_b))
    out_dir = args.out_dir or osp.join(
        'demo/diffusion/samples/comparisons', f'{label_a}_vs_{label_b}'
    )
    os.makedirs(out_dir, exist_ok=True)

    run_a = resolve_run_dir(args.dir_a)
    run_b = resolve_run_dir(args.dir_b)

    pairs = paired_indices(run_a['renders_dir'], run_b['renders_dir'])
    if args.num_samples is not None:
        pairs = pairs[:args.num_samples]
    print(f'[compare_samples_vis] 共 {len(pairs)} 对样本参与对比')

    if not args.skip_paired_gifs:
        gif_dir = osp.join(out_dir, 'paired_gifs')
        for idx, path_a, path_b in pairs:
            out_path = osp.join(gif_dir, f'{idx:05d}_compare.gif')
            make_paired_gif(path_a, path_b, label_a, label_b, out_path, args.fps)
        print(f'[compare_samples_vis] 已保存 {len(pairs)} 个并排对比 GIF 到 {gif_dir}')

    if not args.skip_contact_sheet:
        frame_index = args.frame_index
        if frame_index is None:
            sample_frames = imageio.mimread(pairs[0][1])
            frame_index = len(sample_frames) // 2
        build_contact_sheet(
            pairs, label_a, label_b, out_dir, frame_index,
            tuple(args.thumb_size), args.cols, args.samples_per_page,
        )

    print(
        '[compare_samples_vis] 提示：两组都是无条件、无固定随机种子的采样，'
        '按 index 配对的两个样本不是同一个底层输入的结果，只做批量分布层面的直观对比。'
    )


if __name__ == '__main__':
    main()
