"""Step 8: build datasets/processed/CHI3D/{train,val}_diffusion.pkl.

llib/data/preprocess/chi3d.py already knows how to do this -- CHI3D.load() falls
back to rebuilding the file from datasets/original/CHI3D whenever it is missing
(chi3d.py:426-454). But that fallback only fires the first time a training run
touches the dataset, which means the first thing a multi-hour job does is spend
several minutes doing preprocessing, with a CUDA dependency (chi3d.py:353-354)
and no separate log. Doing it here instead keeps stage 1 self-contained and lets
s09 verify the result before any training starts.

Nothing is reimplemented: this instantiates the production class with the
settings the cond_bev config uses and calls load().

  load_single_camera=False        all 4 camera views, as the paper does for the
                                  conditional model ("we use all camera views of
                                  the MoCap datasets, i.e. 4/8 cameras for
                                  CHI3D/Hi4D")
  load_contact_frame_only=True    one frame per sequence, the annotated contact
                                  frame; 247 for s02+s04
  load_unit_glob_and_transl=False keep the camera-frame target (global_orient_cam
                                  / transl_cam). This is the frame BEV reports
                                  in, so it is the only one where the BEV
                                  conditioning actually determines the target.
                                  Setting it True would also silently collapse
                                  the 4 views to 1 (chi3d.py:456).

Skips a split whose output already exists; delete the file to force a rebuild
(and do delete it after touching train_val_split.npz -- the pkl is a cache and
will not notice the split changed).
"""

import argparse
import os.path as osp

from loguru import logger as guru


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--imar-tools-folder', default='essentials/imar_vision_datasets_tools')
    ap.add_argument('--image-folder', default='images_contact',
                    help='where s01 extracted the contact frames. Only used to read '
                         'image size; chi3d.py:361-366 falls back to a 900x900 zero '
                         'image when a frame is absent, which happens to match CHI3D, '
                         'but pointing at the real frames is the honest thing to do.')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--force', action='store_true', help='rebuild even if the pkl exists')
    args = ap.parse_args()

    # imported here so --help works without a GPU / torch import cost
    from llib.data.preprocess.chi3d import CHI3D

    for split in args.splits:
        out_fn = osp.join(args.processed_data_folder, f'{split}_diffusion.pkl')
        if osp.exists(out_fn) and not args.force:
            guru.info(f'{out_fn} exists, skipping (use --force to rebuild)')
            continue

        guru.info(f'building {out_fn} ...')
        data = CHI3D(
            original_data_folder=args.original_data_folder,
            processed_data_folder=args.processed_data_folder,
            imar_vision_datasets_tools_folder=args.imar_tools_folder,
            image_folder=args.image_folder,
            split=split,
            body_model_type='smplx',
            load_single_camera=False,
            load_from_scratch_single_camera=False,
            load_contact_frame_only=True,
            load_unit_glob_and_transl=False,
        ).load(processed_fn_ext='_diffusion.pkl')
        guru.info(f'wrote {out_fn} ({len(data)} samples after filtering)')


if __name__ == '__main__':
    main()
