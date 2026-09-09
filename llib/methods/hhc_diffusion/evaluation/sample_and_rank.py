"""
Generate a pool of unconditional two-person samples from a trained BUDDI
checkpoint, rank them by inter-person distance (the same proximity signal
used to curate the InterX training data), and export the closest ("most
intimate") or farthest ones as 360-degree turntable GIFs and optionally as
colored .glb 3D files you can open in Blender / a VS Code 3D preview / etc.

This is NOT true conditional generation -- the model is unconditional
(guidance_params=[]), so there's no way to directly ask for "closer"
interactions. This script just generates more samples than you need and
keeps the ones that happen to be closest, which is an honest way to bias
what you look at without pretending the model was steered.

Usage:
    python llib/methods/hhc_diffusion/evaluation/sample_and_rank.py \
        --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
        --checkpoint-name /absolute/path/to/checkpoint.pt \
        --output-folder demo/diffusion/samples/my_run \
        --num-samples 64 --top-k 8 --rank closest --export-glb
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np
import torch
import smplx
import trimesh
import imageio

from llib.defaults.main import config as default_config, merge as merge_configs
from llib.methods.hhc_diffusion.evaluation.utils import setup_diffusion_module
from llib.methods.hhc_diffusion.evaluation.sample import run_sampling
from llib.visualization.scripts.tools import build_renderer, render_360_views

# distinct colors per person (0-255 RGB), matches the palette style used
# elsewhere in llib/visualization/colors.txt
PERSON_COLORS_255 = [[255, 120, 110], [80, 190, 210]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-cfg", dest="exp_cfgs", nargs="+", default=None, required=True,
                         help="Same config used to train the checkpoint, e.g. config_buddi_v02.yaml")
    parser.add_argument("--exp-opts", dest="exp_opts", nargs="*", default=[])
    parser.add_argument("--checkpoint-name", required=True,
                         help="Absolute path to a specific .pt checkpoint file (not 'latest' -- "
                              "this script doesn't try to auto-discover your training run's folder).")
    parser.add_argument("--output-folder", required=True)

    parser.add_argument("--num-samples", type=int, default=64,
                         help="pool size to generate and rank; bigger pool = better chance of finding extreme cases")
    parser.add_argument("--top-k", type=int, default=8,
                         help="how many of the ranked samples to actually render/export")
    parser.add_argument("--rank", choices=["closest", "farthest"], default="closest",
                         help="closest = most 'intimate' (smallest inter-person distance)")

    parser.add_argument("--max-t", type=int, default=1000)
    parser.add_argument("--skip-steps", type=int, default=20,
                         help="bigger = faster/coarser denoising, smaller = slower/higher quality")
    parser.add_argument("--num-poses", type=int, default=24, help="frames per 360-degree GIF")
    parser.add_argument("--render-width", type=int, default=300)
    parser.add_argument("--render-height", type=int, default=300)
    parser.add_argument("--export-glb", action="store_true", default=False,
                         help="also export each kept sample as a colored .glb (open in Blender etc.)")
    parser.add_argument("--seed", type=int, default=None)

    cmd_args = parser.parse_args()

    # run_sampling()/setup_diffusion_module() are shared with sample.py and
    # expect these attributes to exist even though this script only does
    # plain unconditional generation (no inpainting/conditioning).
    cmd_args.inpaint = False
    cmd_args.condition = False
    cmd_args.inpaint_params = []
    cmd_args.inpaint_item_idx = 0
    cmd_args.batch_size = cmd_args.num_samples
    cmd_args.eta = 0.0
    cmd_args.log_steps = max(cmd_args.max_t, 1)  # we only use x_starts["final"], not the intermediate log

    cfg = merge_configs(cmd_args, default_config)
    cfg.batch_size = cmd_args.batch_size
    # point the Logger at our own output folder instead of letting it guess
    # one from the config file path (that guess doesn't know about your
    # actual training run and would just create clutter elsewhere).
    cfg.logging.base_folder = cmd_args.output_folder
    cfg.logging.run = "sample_and_rank"

    return cfg, cmd_args


@torch.no_grad()
def main():
    cfg, cmd_args = parse_args()
    if cmd_args.seed is not None:
        torch.manual_seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)

    os.makedirs(cmd_args.output_folder, exist_ok=True)

    diffusion_module = setup_diffusion_module(cfg, cmd_args)
    _x_ts, x_starts = run_sampling(cfg, cmd_args, diffusion_module, eta=cmd_args.eta)

    final = {k: v.detach().cpu() for k, v in x_starts["final"].items()}
    transl = final["transl"]      # [N, 2, 3]
    verts = final["vertices"]     # [N, 2, V, 3]

    dist = (transl[:, 0] - transl[:, 1]).norm(dim=-1)  # [N]
    order = torch.argsort(dist, descending=(cmd_args.rank == "farthest"))
    top_idx = order[: cmd_args.top_k].tolist()

    with open(osp.join(cmd_args.output_folder, "pool_x_starts_smplx.pkl"), "wb") as f:
        pickle.dump(final, f)
    with open(osp.join(cmd_args.output_folder, "pool_distances.pkl"), "wb") as f:
        pickle.dump(dist.numpy(), f)

    body_model = smplx.create(
        model_path="essentials/body_models", model_type="smplx",
        gender="neutral", num_betas=10, batch_size=2,
    )
    faces_np = body_model.faces
    faces_t = torch.from_numpy(faces_np.astype(np.int64))

    renderer = build_renderer(width=cmd_args.render_width, height=cmd_args.render_height)

    out_dir = osp.join(cmd_args.output_folder, f"top{cmd_args.top_k}_{cmd_args.rank}")
    os.makedirs(out_dir, exist_ok=True)

    for rank, idx in enumerate(top_idx):
        d = dist[idx].item()
        tag = f"rank{rank:02d}_dist{d:.2f}"

        frames = render_360_views(renderer, verts[idx], faces_t, num_poses=cmd_args.num_poses)
        imageio.mimwrite(osp.join(out_dir, f"{tag}.gif"), frames, fps=6)

        if cmd_args.export_glb:
            meshes = [
                trimesh.Trimesh(
                    vertices=verts[idx, h].numpy(), faces=faces_np,
                    vertex_colors=PERSON_COLORS_255[h] + [255], process=False,
                )
                for h in range(2)
            ]
            trimesh.Scene(meshes).export(osp.join(out_dir, f"{tag}.glb"))

    print(
        f"Generated {cmd_args.num_samples} samples, inter-person distance range "
        f"[{dist.min():.3f}, {dist.max():.3f}] meters. "
        f"Saved top {cmd_args.top_k} ({cmd_args.rank}) to {out_dir}"
    )


if __name__ == "__main__":
    main()
